from pydantic import BaseModel


class ModelArchitectureSchema(BaseModel):
    name: str
    weights: str | None


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
