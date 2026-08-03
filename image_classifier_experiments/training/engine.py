import copy
import math
from typing import Literal

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm


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


def train_model(
    model: nn.Module,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    optimizer: Optimizer,
    loss_fn: nn.Module,
    accuracy_fn,
    epochs: int = 3,
    device: torch.device = None,
):
    """Trains a model using CrossEntropyLoss and StochasticGradientDescent with given configuration"""

    # Results and checkpoint tracking
    best_epoch = None
    best_test_acc = float("-inf")
    best_test_loss = None
    best_state_dict = None

    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

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

        # Update results and best checkpoint tracking
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

        # Accuracy will be our measure for best- so "best[metric]" here really
        # means- [metric] associated with best accuracy. Want to make this clear
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_test_loss = test_loss
            best_epoch = epoch
            # MAKE SURE this is a deep copy- not just a copy of the reference which
            # always poitns to the most current state of the model's state_dict
            best_state_dict = copy.deepcopy(model.state_dict())

        if epochs > 10:
            if epoch % 10 == 0:
                tqdm.write(
                    f"Epoch: {epoch} -- Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}\n"
                )
        else:
            tqdm.write(
                f"Epoch: {epoch} -- Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}\n"
            )
    # Ensure training ran and we have a valid trained checkpoint
    if best_state_dict is None or best_epoch is None:
        raise RuntimeError(
            "Training completed without producing a checkpoint. Check that epochs > 0."
        )

    train_results = {
        "history": results,
        "best_checkpoint": {
            "epoch": best_epoch,
            "test_acc": best_test_acc,
            "test_loss": best_test_loss,
            "state_dict": best_state_dict,
        },
    }
    return train_results
