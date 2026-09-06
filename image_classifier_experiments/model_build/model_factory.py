from torch import nn
from torchvision import models


class BackboneFactory:
    @staticmethod
    def build_backbone(model_arch_name: str, pretrained: bool = True):

        model_name = model_arch_name.lower()

        try:
            factory_fn = getattr(models, model_name)
        except AttributeError as e:
            raise ValueError(
                f"No torchvision model found for architectue name: {model_name}. {str(e)}"
            )

        weights = None
        if pretrained:
            try:
                weights_enum = models.get_model_weights(model_name)
                weights = weights_enum.DEFAULT
            except ValueError as e:
                raise ValueError(
                    f"No default pre-trained weights found for architecture name: {model_name}"
                )

        return factory_fn(weights=weights), weights
