"""
A series of helper functions used throughout the course.

If a function gets defined once and could be used over and over, it'll go in here.
"""

import io
from pathlib import Path
from typing import Any

import boto3
import matplotlib.pyplot as plt
import torch


# Plot linear data or training and test and predictions (optional)
def plot_predictions(
    train_data, train_labels, test_data, test_labels, predictions=None
):
    """
    Plots linear training data and test data and compares predictions.
    """
    plt.figure(figsize=(10, 7))

    # Plot training data in blue
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")

    # Plot test data in green
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

    if predictions is not None:
        # Plot the predictions in red (predictions were made on the test data)
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    # Show the legend
    plt.legend(prop={"size": 14})


# Calculate accuracy (a classification metric)
def accuracy_fn(y_true, y_pred):
    """Calculates accuracy between truth labels and predictions.

    Args:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared to predictions.

    Returns:
        [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
    """
    correct = torch.eq(y_true, y_pred).sum()
    acc = (correct / len(y_pred)) * 100
    return acc


def print_train_time(start, end, device=None):
    """Prints difference between start and end time.

    Args:
        start (float): Start time of computation (preferred in timeit format).
        end (float): End time of computation.
        device ([type], optional): Device that compute is running on. Defaults to None.

    Returns:
        float: time between start and end in seconds (higher is longer).
    """
    total_time = end - start
    print(f"\nTrain time on {device}: {total_time:.3f} seconds")
    return total_time


def save_model(model: torch.nn.Module, target_dir: str, model_name: str):
    """Saves a PyTorch model to a target directory.

    Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

    Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    # Create model save path
    assert model_name.endswith((".pth", ".pt")), (
        "model_name should end with '.pt' or '.pth'"
    )
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)


def save_model_checkpoint(
    state_dict: dict[str, Any],
    model_metadata: dict[str, Any],
    target_dir: str,
    file_name: str,
):
    """Saves a PyTorch model and relevant checkpoint metadata to target directory"""
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    # Create model save path
    assert file_name.endswith((".pth", ".pt")), (
        "model_name should end with '.pt' or '.pth'"
    )
    model_save_path = target_dir_path / file_name

    # debug printing
    model_checkpoint = {
        "state_dict": state_dict,
        "metadata": model_metadata,
    }
    checkpoint_debug = {k: v for k, v in model_checkpoint.items() if k != "state_dict"}
    print(
        f"[INFO] Saving model checkpoint with metadata:  {checkpoint_debug} to: {model_save_path}"
    )
    # print(model_checkpoint, file=open("output.txt", "w"))

    torch.save(
        obj=model_checkpoint,
        f=model_save_path,
    )


def save_model_checkpoint_s3(
    state_dict: dict[str, Any],
    model_metadata: dict[str, Any],
    bucket_name: str,
    object_key: str,
):
    """Saves a PyTorch model and relevant checkpoint metadta to S3"""

    assert object_key.endswith((".pth", ".pt")), (
        "model_name should end with '.pt' or '.pth'"
    )

    s3 = boto3.client("s3")

    model_checkpoint = {
        "state_dict": state_dict,
        "metadata": model_metadata,
    }

    with io.BytesIO() as buffer:
        torch.save(
            obj=model_checkpoint,
            f=buffer,
        )

        s3.put_object(
            Bucket=bucket_name,
            Key=object_key,
            Body=buffer.getvalue(),
        )

    # debug printing
    checkpoint_debug = {k: v for k, v in model_checkpoint.items() if k != "state_dict"}
    print(
        f"[INFO] Saving model checkpoint with metadata:  {checkpoint_debug} to: {bucket_name}/{object_key}"
    )


def plot_loss_curves(results: dict[str, list[float]]):
    loss = results["train_loss"]
    test_loss = results["test_loss"]

    acc = results["train_acc"]
    test_acc = results["test_acc"]

    # Get number of epochs (using loss, but they should all have the same length)
    epochs = range(len(loss))

    plt.figure(figsize=(15, 7))

    # Plot the loss curves
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, acc, label="train accuracy")
    plt.plot(epochs, test_acc, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()

    plt.show()


# Helper debug funcrion for validating training checkpoint data
# without printing out the huge state dict values
def print_checkpoint_values(chkpt):
    # 1. Safely extract a dictionary map of properties from your custom object
    if isinstance(chkpt, dict):
        chkpt_dict = chkpt
    else:
        try:
            chkpt_dict = vars(chkpt)
        except TypeError:
            # Fallback if the class doesn't support vars()
            chkpt_dict = {
                attr: getattr(chkpt, attr)
                for attr in dir(chkpt)
                if not attr.startswith("__")
            }

    # 2. Iterate through the discovered state dicts
    for dict_name, s_dict in chkpt_dict.items():
        # Only process actual dictionaries (like model, optimizer, or scheduler states)
        if not isinstance(s_dict, dict):
            continue

        print(f"\n=== 📦 {dict_name.upper()} ===")
        for k, v in s_dict.items():
            # Handle regular layer tensors (Model weights)
            if hasattr(v, "shape"):
                print(f"  {k:<35} -> Shape: {list(v.shape)}")

            # Handle the nested Optimizer tracker tensors safely
            elif k == "state" and isinstance(v, dict):
                sample_id = next(iter(v)) if v else None
                sample_shapes = (
                    [
                        f"{sk}: {list(sv.shape)}"
                        for sk, sv in v[sample_id].items()
                        if hasattr(sv, "shape")
                    ]
                    if sample_id
                    else []
                )
                print(
                    f"  {k:<35} -> Dict tracking {len(v)} weights. (e.g., Param {sample_id} tracking shapes: {sample_shapes})"
                )

            # Handle Optimizer Learning Rates & Hyperparameters
            elif k == "param_groups" and isinstance(v, list):
                clean_groups = [
                    {
                        g_key: g_val
                        for g_key, g_val in group.items()
                        if g_key != "params"
                    }
                    for group in v
                ]
                print(f"  {k:<35} -> Config Values: {clean_groups}")

            # Handle Scheduler configurations and all other fallback values
            else:
                print(f"  {k:<35} -> Value: {v}")


def matches_state_dict(source_dict, match_dict) -> bool:
    """
    Compares two PyTorch state_dicts (Model, Optimizer, or LR Scheduler).
    Handles mixed types like Tensors, nested dicts, lists, and primitives.
    """
    # 1. Quick reference check
    if source_dict is match_dict:
        return True

    # 2. Check if types match
    if type(source_dict) is not type(match_dict):
        return False

    # 3. Handle dictionaries (recursive check)
    if isinstance(source_dict, dict):
        if set(source_dict.keys()) != set(match_dict.keys()):
            return False
        for key in source_dict:
            if not matches_state_dict(source_dict[key], match_dict[key]):
                return False
        return True

    # 4. Handle lists/tuples (recursive check)
    if isinstance(source_dict, (list, tuple)):
        if len(source_dict) != len(match_dict):
            return False
        for item1, item2 in zip(source_dict, match_dict):
            if not matches_state_dict(item1, item2):
                return False
        return True

    # 5. Handle PyTorch Tensors
    if isinstance(source_dict, torch.Tensor):
        # Cast match_dict to source_dict's device purely for the evaluation step.
        # the loaded state dicts will reconcile to the correct training device
        # outside of here, so for these purposes we can ignore device mitmatch
        # between the resume_checkpoint state_dicts which are moved to CPU for
        # initial load.
        if source_dict.device != match_dict.device:
            match_dict = match_dict.to(source_dict.device)

        if (
            source_dict.shape != match_dict.shape
            or source_dict.dtype != match_dict.dtype
        ):
            return False
        return torch.equal(source_dict, match_dict)

    # 6. Handle standard Python primitives (int, float, str, bool, None)
    return source_dict == match_dict
