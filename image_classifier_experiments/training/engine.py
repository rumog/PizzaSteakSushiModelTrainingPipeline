import copy
import math
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def get_current_lr(optimizer):
    return optimizer.param_groups[0]["lr"]


def train_step(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: Optimizer,
    accuracy_fn,
    device: torch.device = None,
):

    # 0. Put model into training mode
    model.train()

    train_loss, train_acc = 0, 0

    # For each batch
    for batch, (X_batch, y_batch) in enumerate(dataloader):
        # Move data to device
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        # 1. Forward Pass (don't forget inference mode!)
        y_logits = model(X_batch)

        # 2. calculate loss and accuracy
        loss = loss_fn(y_logits, y_batch)
        acc = accuracy_fn(y_true=y_batch, y_pred=y_logits.argmax(dim=1))

        # keeps only the value, not the whole tensor (which is attached to device, grad info, etc).  Earlier
        # tutorial notebooks omitted this, but now we're using it.  You'll see if you avoid this step you can
        # run into issues tying to do things like plot the result as matplotlib will complain that it can't use cuda/mps
        # tensors etc.  Doing this just makes processing results easier so you don't have to convert later every time
        # something doesn't want a torch tensor.
        train_loss += loss.item()
        train_acc += acc

        # 3. Zero gradient
        optimizer.zero_grad()

        # 4. Loss Backward / Backpropagation
        loss.backward()

        # 5. Optimizer Step
        optimizer.step()

    # Calculate results: the avg train loss and accuracy metric per batch
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    return train_loss, train_acc
    # print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")


def test_step(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    accuracy_fn,
    device: torch.device = None,
):

    # Set model to eval mode
    model.eval()
    test_loss, test_acc = 0, 0

    # For each batch
    for batch, (X_batch, y_batch) in enumerate(dataloader):
        # 0. Move data to device
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        # 1. Forward Pass
        with torch.inference_mode():
            y_logits = model(X_batch)

        # 2. Calculate loss and accuracy
        loss = loss_fn(y_logits, y_batch)
        acc = accuracy_fn(y_true=y_batch, y_pred=y_logits.argmax(dim=1))

        test_loss += loss.item()
        test_acc += acc

    # Calculate results: the avg test loss and accuracy metric per batch
    test_loss /= len(dataloader)
    test_acc /= len(dataloader)
    return test_loss, test_acc


# Keeping a note here on patience, move this somewhre when yu get a chance
# Learning Rate Schedules: Early stopping patience must always be longer than your Learning Rate (LR)
# scheduler's patience or decay frequency. If you use a ReduceLROnPlateau scheduler with a patience of 3,
# your early stopping patience should be around 7–10 epochs to give the
# model time to stabilize and find a better local minimum at the lower learning rate.


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
    writer: SummaryWriter | None = None,
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
    }

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            accuracy_fn=accuracy_fn,
            device=device,
        )
        test_loss, test_acc = test_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            accuracy_fn=accuracy_fn,
            device=device,
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

        # Update results and best checkpoint tracking
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        results["lr"].append(epoch_lr)

        # Also update writer for TensorBoard integration
        if writer is not None:
            # Plot Test vs Train on same graph for loss/acc
            writer.add_scalars(
                "Loss Comparison",
                {
                    "train": train_loss,
                    "test": test_loss,
                },
                epoch,
            )

            writer.add_scalars(
                "Accuracy Comparison",
                {
                    "train": train_acc,
                    "test": test_acc,
                },
                epoch,
            )

            # plot individual graphs
            writer.add_scalar(
                "Loss/train",
                train_loss,
                epoch,
            )
            writer.add_scalar(
                "Loss/test",
                test_loss,
                epoch,
            )
            writer.add_scalar(
                "Accuracy/train",
                train_acc,
                epoch,
            )
            writer.add_scalar(
                "Accuracy/test",
                test_acc,
                epoch,
            )
            writer.add_scalar(
                "Learning Rate",
                epoch_lr,
                epoch,
            )

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
                    f"Epoch: {epoch} -- LR: {epoch_lr} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f} | Epochs w/o accuracy impr: {epochs_without_improvement}\n"
                )
        else:
            tqdm.write(
                f"Epoch: {epoch} -- LR: {epoch_lr} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f} Epochs w/o accuracy impr: {epochs_without_improvement}\n"
            )

        # stop early if epochs without improvement breaches patience
        if patience is not None:
            tqdm.write(
                f"early stop patience set to {patience}, checking early stop conditions"
            )
            if epochs_without_improvement >= patience:
                tqdm.write(
                    f"patience value: {patience} was supplied, checking for early stoppage"
                )
                tqdm.write(
                    f"Early stopping triggered at epoch {epoch}. Best epoch: {best_epoch}"
                )
                break

        # prepare next epoch
        if scheduler is not None:
            tqdm.write("scheduler was provided, stepping scheduler")
            scheduler.step(test_loss)

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
