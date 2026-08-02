import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def eval_model(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    accuracy_fn,
    device: torch.device = None,
):
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in tqdm(data_loader):
            X, y = X.to(device), y.to(device)
            # 1. Forward Pass
            y_logits = model(X)

            # 2. Calculate loss and accuracy
            loss += loss_fn(y_logits, y)
            acc += accuracy_fn(y_true=y, y_pred=y_logits.argmax(dim=1))

        # Scale l oss and acc to find the average loss/acc per batch
        loss /= len(data_loader)
        acc /= len(data_loader)

    return {
        "model_name": model.__class__.__name__,  # only works when model was created with a class
        "model_loss": loss.item(),
        "model_acc": acc,
    }
