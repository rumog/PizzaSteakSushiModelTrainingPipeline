import copy
import math
from collections.abc import Callable
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb


def get_optimizer_group_metrics(optimizer: torch.optim.Optimizer):

    metrics = {}
    for i, group in enumerate(optimizer.param_groups):
        metrics[f"optimizer/group {i}/lr"] = group["lr"]
        metrics[f"optimizer/group {i}/wd"] = group["weight_decay"]
    return metrics


def get_current_lr(optimizer):
    return optimizer.param_groups[0]["lr"]


def get_backbone_lr(optimizer):
    if len(optimizer.param_groups) > 1:
        return optimizer.param_groups[1]["lr"]
    else:
        return None


def get_backbone_lrs_string(optimizer: torch.optim.Optimizer):
    lr_strings = []

    # By current design, group 0 is always the classifier params
    # groups for indices > 0 are assoiated with each respective
    # stage of unfrozen backbone layers
    if not len(optimizer.param_groups) > 1:
        return "None"

    for i, group in enumerate(optimizer.param_groups[1:], start=1):
        lr_strings.append(f"{i}:{group['lr']:.7f}")

    if lr_strings:
        return ",".join(lr_strings)
    else:
        return "None"


def get_backbone_wds_string(optimizer: torch.optim.Optimizer):
    lr_strings = []

    # By current design, group 0 is always the classifier params
    # groups for indices > 0 are assoiated with each respective
    # stage of unfrozen backbone layers
    if not len(optimizer.param_groups) > 1:
        return "None"

    for i, group in enumerate(optimizer.param_groups[1:], start=1):
        lr_strings.append(f"{i}:{group['weight_decay']:.7f}")

    if lr_strings:
        return ",".join(lr_strings)
    else:
        return "None"


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

    torch.save(
        {
            "features": features,
            "labels": labels,
        },
        output_dir,
    )


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
):
    """Trains a model using CrossEntropyLoss and StochasticGradientDescent with given configuration"""

    # Results and checkpoint tracking
    best_epoch = None
    best_test_acc = float("-inf")
    best_test_loss = None
    best_state_dict = None
    best_lr = None

    epochs_without_improvement = 0
    epochs_completed = 0

    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "lr": [],
        "backbone_lr": [],
    }

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

    for epoch in tqdm(range(epochs), desc="Executing model training..."):
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
        epoch_lr = get_current_lr(optimizer)
        epoch_backbone_lr = get_backbone_lr(optimizer)

        # Update results and best checkpoint tracking
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        results["lr"].append(epoch_lr)
        if epoch_backbone_lr is not None:
            results["backbone_lr"].append(epoch_backbone_lr)

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
            wandb.log(metrics, step=epoch + 1)

        # Accuracy will be our measure for best- so "best[metric]" here really
        # means- [metric] associated with best accuracy. Want to make this clear
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_test_loss = test_loss
            best_epoch = epoch
            best_lr = epoch_lr
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
                    f"Epoch: {epoch} -- Classifier LR: {epoch_lr} | -- BB LR: [{get_backbone_lrs_string(optimizer)}] | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f} | Epochs w/o accuracy impr: {epochs_without_improvement}\n"
                )
        else:
            tqdm.write(
                f"Epoch: {epoch} -- Classifier LR: {epoch_lr} | -- BB LR: [{get_backbone_lrs_string(optimizer)}] | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f} Epochs w/o accuracy impr: {epochs_without_improvement}\n"
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
            "epoch_lr": best_lr,
            "test_acc": best_test_acc,
            "test_loss": best_test_loss,
            "state_dict": best_state_dict,
        },
    }
    return train_results
