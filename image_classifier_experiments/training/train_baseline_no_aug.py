from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import torch
from torch import nn
from torchvision import transforms

from image_classifier_experiments.data_setup import data_setup
from image_classifier_experiments.model_build.tiny_vgg_baseline import TinyVGGBaseline
from image_classifier_experiments.training import engine
from image_classifier_experiments.utils.helper_functions import (
    accuracy_fn,
    plot_loss_curves,
    save_model_checkpoint,
)

EPOCHS = 50
BATCH_SIZE = 32
NUM_WORKERS = 0  # os.cpu_count()
DATA_PATH_PARENT_DIR = "data/"
IMAGE_PATH_PARENT_DIR = "pizza_steak_sushi"
MODEL_SAVE_DIR = "model"
LEARNING_RATE = 0.001
IMAGE_SIZE = (64, 64)

# model hyperparams
INPUT_UNITS = 3
HIDDEN_LAYER_UNITS = 10

torch.manual_seed(42)
torch.mps.manual_seed(42)

data_path = Path(DATA_PATH_PARENT_DIR)
image_path = data_path / IMAGE_PATH_PARENT_DIR
train_dir = image_path / "train"
test_dir = image_path / "test"

# Hard code device for compatibility with M1 mac for now
# change this if expanding.
device = "mps" if torch.mps.is_available() else "cpu"

# Create a simple transform with minimal augmentaiton.
# Can expand on this with a collection of common transforms used
# for experimentation.
simple_transform = transforms.Compose(
    [transforms.Resize(size=IMAGE_SIZE), transforms.ToTensor()]
)

simple_train_dataloader, simple_test_dataloader, class_list = (
    data_setup.create_dataloaders(
        train_dir,
        test_dir,
        simple_transform,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle_train=True,
    )
)

model_0 = TinyVGGBaseline(
    in_channels=INPUT_UNITS,
    hidden_units=HIDDEN_LAYER_UNITS,
    out_channels=len(class_list),
).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_0.parameters(), lr=LEARNING_RATE)

results = engine.train_model(
    model=model_0,
    train_dataloader=simple_train_dataloader,
    test_dataloader=simple_test_dataloader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    accuracy_fn=accuracy_fn,
    epochs=EPOCHS,
    device=device,
)

# Create model save
timestamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y%m%d_%H%M%S")

filename = f"{model_0.__class__.__name__}_{timestamp}.pth"

# MODEL_NAME = "intro_pytorch_computer_vision_model_2.pth"
model_architecture = {
    "name": filename,
    "in_units": INPUT_UNITS,
    "hidden_layer_units": HIDDEN_LAYER_UNITS,
    "out_units": len(class_list),
}
save_model_checkpoint(
    model_0, model_architecture, class_list, IMAGE_SIZE, MODEL_SAVE_DIR, filename
)
plot_loss_curves(results)
