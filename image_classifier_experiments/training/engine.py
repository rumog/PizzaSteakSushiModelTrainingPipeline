import copy
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from image_classifier_experiments.training.checkpoint.types.training_checkpoint import (
    TrainingCheckpointMetadataSchema,
)
from image_classifier_experiments.training.checkpoint.types.training_checkpoint_data import (
    TrainCheckpoint,
)

CHECKPOINT_DIR = "checkpoint"


def get_optimizer_group_metrics(optimizer: torch.optim.Optimizer):
    metrics = {}

    for i, group in enumerate(optimizer.param_groups):
        metrics[f"optimizer/group {i}/lr"] = group["lr"]
        metrics[f"optimizer/group {i}/wd"] = group["weight_decay"]
    return metrics


def get_optimizer_groups_config(optimizer: torch.optim.Optimizer, config_key: str):
    param_group_configs = []
    if not config_key:
        return param_group_configs
    for group in optimizer.param_groups:
        param_group_configs.append(group[config_key])
    return param_group_configs


def print_optimizier_group_metrics(optimizer: torch.optim.Optimizer):
    lr_strings = []
    wd_strings = []

    for i, group in enumerate(optimizer.param_groups):
        lr_strings.append(f"[{group['lr']:.7g}]")
        wd_strings.append(f"[{group['weight_decay']:.7g}]")

    lr_string = f"LR: {','.join(lr_strings)}" if lr_strings else "LR: [None]"
    wd_string = f"WD: {','.join(wd_strings)}" if wd_strings else "WD: [None]"

    return f"{lr_string} | {wd_string}"


def extract_backbone_features(
    backbone: nn.Module, dataloader: DataLoader, device: torch.device, output_dir: str
):
    backbone.eval()

    all_features = []
    all_labels = []

    with torch.inference_mode():
        for X, y in tqdm(
            dataloader, desc="Extracting backbone features for dataloader..."
        ):
            X = X.to(device)

            features = backbone(X)

            all_features.append(features.cpu())
            all_labels.append(y)

    features = torch.cat(all_features)
    labels = torch.cat(all_labels)
    print(f"Saving cached features to: {output_dir}")
    torch.save(
        {
            "features": features,
            "labels": labels,
        },
        output_dir,
    )


def save_training_checkpoint_artifact(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LRScheduler | ReduceLROnPlateau,
    completed_epochs: int,
    scheduled_epochs: int,
    best_checkpoint: dict[str, Any],
    epochs_without_improvement: int,
    history: dict[str, Any],
    target_dir: str = ".",
    file_name: str = "checkpoint_latest.pth",
):
    target_dir_path = Path(target_dir) / CHECKPOINT_DIR
    target_dir_path.mkdir(parents=True, exist_ok=True)

    # Create model save path
    assert file_name.endswith((".pth", ".pt")), (
        "model_name should end with '.pt' or '.pth'"
    )
    checkpoint_save_path = target_dir_path / file_name

    model_state_dict = copy.deepcopy(model.state_dict())
    optimizer_state_dict = copy.deepcopy(optimizer.state_dict())
    scheduler_state_dict = copy.deepcopy(scheduler.state_dict())

    best_checkpoint_dict = {
        "epoch": best_checkpoint.get("epoch"),
        "test_acc": best_checkpoint.get("test_acc"),
        "test_loss": best_checkpoint.get("test_loss"),
        "state_dict": best_checkpoint.get("state_dict"),
    }

    train_metadata = {
        "history": history,
        "last_epoch": completed_epochs,
        "scheduled_epochs": scheduled_epochs,
        "epochs_without_improvement": epochs_without_improvement,
        "best_epoch_checkpoint": best_checkpoint_dict,
    }

    # validate checkpoint metadata expected schema and
    # save dump from schema object to ensure consistent save structure
    checkpoint_metadata_schema = TrainingCheckpointMetadataSchema.model_validate(
        train_metadata
    )
    validated_train_metadata = checkpoint_metadata_schema.model_dump()

    training_checkpoint = {
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer_state_dict,
        "scheduler_state_dict": scheduler_state_dict,
        "metadata": validated_train_metadata,
        # - what happens to a wb run?  If you name it the same will it just resume? if so wb config
    }

    # write to temp save path then swap so most recent checkpoint won't get corrupted if
    # interrupted during a save
    temp_save_path = checkpoint_save_path.with_suffix(".tmp")

    torch.save(
        obj=training_checkpoint,
        f=temp_save_path,
    )
    temp_save_path.replace(checkpoint_save_path)


def train_step(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: Optimizer,
    accuracy_fn,
    device: torch.device = None,
    gpu_transform: Callable = None,
    ram_caching_enabled: bool = False,
):

    # 0. Put model into training mode
    model.train()
    # below is for debugging, only works for top level model, not during backbone caching
    # where only classifier is passed in to train step
    # model.print_training_modes()

    train_loss = torch.zeros((), device=device)
    train_acc = torch.zeros((), device=device)

    # For each batch
    for batch, (X_batch, y_batch) in enumerate(dataloader):
        # Move data to device
        y_batch = y_batch.to(device, non_blocking=True)

        if gpu_transform is not None:
            if ram_caching_enabled:
                X_batch = [
                    gpu_transform(
                        image.to(device, non_blocking=True).float().div_(255.0)
                    )
                    for image in X_batch
                ]
                X_batch = torch.stack(X_batch)
            else:
                X_batch = X_batch.to(device, non_blocking=True)
                X_batch = X_batch.float().div_(255.0)
                X_batch = gpu_transform(X_batch)
        else:
            X_batch = X_batch.to(device, non_blocking=True)

        # 1. Forward Pass
        y_logits = model(X_batch)

        # 2. calculate loss and accuracy
        loss = loss_fn(y_logits, y_batch)
        acc = accuracy_fn(y_true=y_batch, y_pred=y_logits.argmax(dim=1))

        # keeps only the value, not the whole tensor (which is attached to device, grad info, etc).  Earlier
        # tutorial notebooks omitted this, but now we're using it.  You'll see if you avoid this step you can
        # run into issues tying to do things like plot the result as matplotlib will complain that it can't use cuda/mps
        # tensors etc.  Doing this just makes processing results easier so you don't have to convert later every time
        # something doesn't want a torch tensor.
        train_loss += loss.detach()
        train_acc += acc.detach()

        # 3. Zero gradient
        optimizer.zero_grad()

        # 4. Loss Backward / Backpropagation
        loss.backward()

        # 5. Optimizer Step
        optimizer.step()

    # Calculate results: the avg train loss and accuracy metric per batch
    train_loss = (train_loss / len(dataloader)).item()
    train_acc = (train_acc / len(dataloader)).item()
    return train_loss, train_acc
    # print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")


def test_step(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    accuracy_fn,
    device: torch.device = None,
    test_transform: Callable = None,
    ram_caching_enabled: bool = False,
):

    # Set model to eval mode
    model.eval()
    test_loss = torch.zeros((), device=device)
    test_acc = torch.zeros((), device=device)

    # For each batch
    for batch, (X_batch, y_batch) in enumerate(dataloader):
        # 0. Move data to device
        y_batch = y_batch.to(device, non_blocking=True)

        if test_transform is not None and ram_caching_enabled:
            X_batch = [
                test_transform(image.to(device, non_blocking=True).float().div_(255.0))
                for image in X_batch
            ]
            X_batch = torch.stack(X_batch)
        else:
            X_batch = X_batch.to(device, non_blocking=True)

        # 1. Forward Pass
        with torch.inference_mode():
            y_logits = model(X_batch)

        # 2. Calculate loss and accuracy
        loss = loss_fn(y_logits, y_batch)
        acc = accuracy_fn(y_true=y_batch, y_pred=y_logits.argmax(dim=1))

        test_loss += loss.detach()
        test_acc += acc.detach()

    # Calculate results: the avg test loss and accuracy metric per batch
    test_loss = (test_loss / len(dataloader)).item()
    test_acc = (test_acc / len(dataloader)).item()
    return test_loss, test_acc


def train_model(
    model: nn.Module,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    optimizer: Optimizer,
    loss_fn: nn.Module,
    accuracy_fn,
    epochs: int = 3,
    patience: int | None = None,
    scheduler: Any | None = None,
    device: torch.device | None = None,
    # writer: SummaryWriter | None = None,
    caching_enabled: bool = False,
    gpu_transform: Callable = None,
    test_transform: Callable = None,
    ram_caching_enabled: bool = False,
    resume_checkpoint: TrainCheckpoint | None = None,
    run_dir: str | Path = ".",
):
    """Trains a model using CrossEntropyLoss and StochasticGradientDescent with given configuration"""

    # Results and checkpoint tracking
    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "optimizer_group_lr": [[] for _ in range(len(optimizer.param_groups))],
        "optimizer_group_wd": [[] for _ in range(len(optimizer.param_groups))],
    }

    epochs_completed = 0
    epochs_without_improvement = 0

    best_epoch = None
    best_test_acc = float("-inf")
    best_test_loss = None
    best_state_dict = None

    if resume_checkpoint is not None:
        print("Initializing training state from resume checkpoint...")
        results = resume_checkpoint.metadata.history

        epochs_completed = resume_checkpoint.metadata.last_epoch
        epochs_without_improvement = (
            resume_checkpoint.metadata.epochs_without_improvement
        )

        # update results and checkpoint tracking from resume checkpoing
        best_epoch = resume_checkpoint.metadata.best_epoch_checkpoint.epoch
        best_test_acc = resume_checkpoint.metadata.best_epoch_checkpoint.test_acc
        best_test_loss = resume_checkpoint.metadata.best_epoch_checkpoint.test_loss
        best_state_dict = resume_checkpoint.metadata.best_epoch_checkpoint.state_dict

    """
    # Loop only over the remaining epochs, but tell tqdm the full picture
    for epoch in tqdm(
        range(completed_epochs, total_epochs), 
        desc="Executing model training...", 
        initial=completed_epochs, 
        total=total_epochs
    ):
    """
    # If backbone caching is enabled, then we're only training the classifier
    # using cached features dataset, instead of training the whole model using
    # the image dataset.  This drastically reduces train time, with the tradeoff
    # being no random augmentation of data between epochs (feature data is already cached)
    # We can enable or disable backbone caching depending on whether data variation between
    # epochs is being used.
    if caching_enabled:
        train_model = model.classifier
    else:
        train_model = model

    # for epoch in tqdm(range(epochs), desc="Executing model training..."):
    # NOTE: epochs (total epochs) is currently coming from the current training config, not the
    # checkpoint's "scheduled_epochs".
    for epoch in tqdm(
        range(epochs_completed, epochs),
        desc="Executing model training...",
        initial=epochs_completed,
        total=epochs,
    ):
        train_loss, train_acc = train_step(
            model=train_model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            accuracy_fn=accuracy_fn,
            device=device,
            gpu_transform=gpu_transform,
            ram_caching_enabled=ram_caching_enabled,
        )
        test_loss, test_acc = test_step(
            model=train_model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            accuracy_fn=accuracy_fn,
            device=device,
            test_transform=test_transform,
            ram_caching_enabled=ram_caching_enabled,
        )

        # Guard against invalid metrics to prevent saving invalid checkpoints
        if not all(
            math.isfinite(metric)
            for metric in [train_loss, train_acc, test_loss, test_acc]
        ):
            raise RuntimeError(
                f"Invalid metrics encountered. "
                f"Train loss: {train_loss}, Train accuracy: {train_acc}, "
                f"Test loss: {test_loss}, Test accuracy: {test_acc}"
            )

        epochs_completed += 1

        # Update results and best checkpoint tracking
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

        # update lr history for all param groups in optimizer
        for i, lr in enumerate(get_optimizer_groups_config(optimizer, config_key="lr")):
            results["optimizer_group_lr"][i].append(lr)

        # update wd history for all param groups in optimizer
        for i, wd in enumerate(
            get_optimizer_groups_config(optimizer, config_key="weight_decay")
        ):
            results["optimizer_group_wd"][i].append(wd)

        # Also update writer for wandb integration
        if wandb.run is not None:
            metrics = {
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "train/accuracy": train_acc,
                "test/loss": test_loss,
                "test/accuracy": test_acc,
            }
            metrics.update(get_optimizer_group_metrics(optimizer))
            wandb.log(metrics)

        # Accuracy will be our measure for best- so "best[metric]" here really
        # means- [metric] associated with best accuracy. Want to make this clear
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_test_loss = test_loss
            best_epoch = epoch
            # MAKE SURE this is a deep copy- not just a copy of the reference which
            # always poitns to the most current state of the model's state_dict
            best_state_dict = copy.deepcopy(model.state_dict())

            # reset epochs without improvement
            epochs_without_improvement = 0

        else:
            # increment epochs without improvement
            epochs_without_improvement += 1

        if epochs > 50:
            if epoch % 10 == 0:
                tqdm.write(
                    f"Epoch: {epoch} | {print_optimizier_group_metrics(optimizer)} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f} | Epochs w/o accuracy impr: {epochs_without_improvement}\n"
                )
        else:
            tqdm.write(
                f"Epoch: {epoch} | {print_optimizier_group_metrics(optimizer)} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f} Epochs w/o accuracy impr: {epochs_without_improvement}\n"
            )

        # stop early if epochs without improvement breaches patience
        if patience is not None and epochs_without_improvement >= patience:
            tqdm.write(
                f"Early stopping triggered at epoch {epoch}. Best epoch: {best_epoch}"
            )
            break

        # prepare next epoch
        # [TODO]: more hard coding and training loop needing to know implementation details about
        # scheduler- clean this up.
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(test_loss)
            else:
                scheduler.step()

        # Saving training checkpoint for resumable training
        best_checkpoint = {
            "epoch": best_epoch,
            "test_acc": best_test_acc,
            "test_loss": best_test_loss,
            "state_dict": best_state_dict,
        }

        save_training_checkpoint_artifact(
            model=train_model,
            optimizer=optimizer,
            scheduler=scheduler,
            completed_epochs=epochs_completed,
            scheduled_epochs=epochs,
            best_checkpoint=best_checkpoint,
            epochs_without_improvement=epochs_without_improvement,
            history=results,
            target_dir=run_dir,
        )

    # Ensure training ran and we have a valid trained checkpoint
    if best_state_dict is None or best_epoch is None:
        raise RuntimeError(
            "Training completed without producing a checkpoint. Check that epochs > 0."
        )

    train_results = {
        "history": results,
        "train_metadata": {
            "epochs_completed": epochs_completed,
            "epochs_scheduled": epochs,
            "stopped_early": epochs_completed < epochs,
        },
        "best_checkpoint": {
            "epoch": best_epoch,
            "test_acc": best_test_acc,
            "test_loss": best_test_loss,
            "state_dict": best_state_dict,
        },
    }
    return train_results
