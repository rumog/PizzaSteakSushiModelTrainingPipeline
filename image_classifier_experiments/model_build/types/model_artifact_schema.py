from pydantic import BaseModel


class ModelArchitectureSchema(BaseModel):
    name: str
    # This should be required when schema is finalized
    # allowing None for now for backwards compatibility
    backbone: str | None = None
    weights: str | None = None


class ModelPreprocessingSchema(BaseModel):
    image_size: tuple[int, int]


class ModelTrainingInfoSchema(BaseModel):
    epoch: int | None = None
    validation_loss: float | None = None
    validation_accuracy: float | None = None


class ModelMetadataSchema(BaseModel):
    class_list: list[str]
    architecture: ModelArchitectureSchema
    preprocessing: ModelPreprocessingSchema
    training: ModelTrainingInfoSchema | None = None
