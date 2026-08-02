from collections.abc import Callable

import torch
from PIL import Image
from torch import nn
from torchvision import transforms


def predict_image(
    model: nn.Module,
    img: Image.Image,
    class_names: list[str],
    transform: Callable,
    image_size: tuple[int, int] = (64, 64),
    device: torch.device = None,
):
    # [TODO: come back and write error handling]
    # Also consider opening with torch since we're using torch and it avoids manually turning to tensor
    # img = Image.open(image_path)

    # Image transformation if necessary.  We choose to privde a default transform if none provided
    # Note that this makes the image_size parameter kind of confusing/clunky, consider updating this.
    if transform is not None:
        img_transform = transform
    else:
        img_transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                # [TODO]: got this from tutorial, look into how it works
                # Removing for now, this looks like a ResNet standard
                # Used if we're doing transfer learning.  See the note
                # on this src file in under section 5 going modular and section 6
                # transfer learning https://www.learnpytorch.io/06_pytorch_transfer_learning/
                # transforms.Normalize(
                #    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                # ),
            ]
        )

    # Transform and add an extra dimension to image (model requires samples in
    # [batch_size, color_channels, height, width])
    # Also move to device
    img_transformed = img_transform(img).unsqueeze(dim=0).to(device)

    # model setup
    model.to(device)
    model.eval()
    with torch.inference_mode():
        prediction_logits = model(img_transformed)

    prediction_probs = prediction_logits.softmax(dim=1)
    # Remember to use .item() here so the index is a raw integer value and not a cpu/gpu tensor
    # which can break in some scenarios (specifically the GPU tensor scenario)

    # Select the winning class's probability (confidence score)
    prediction_prob = prediction_probs.max().item()

    # Select the winning classs' human readable label
    prediction_label = class_names[prediction_probs.argmax(dim=1).item()]

    return prediction_label, prediction_prob
