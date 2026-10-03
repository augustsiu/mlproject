from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


if __name__ == "__main__":
    # 1. Load and split the raw data
    data_ingestion = DataIngestion()

    train_data_path, test_data_path = (
        data_ingestion.initiate_data_ingestion()
    )

    # 2. Transform the training and test data
    data_transformation = DataTransformation()

    train_array, test_array, _ = (
        data_transformation.initiate_data_transformation(
            train_data_path,
            test_data_path
        )
    )

    # 3. Train several models and save the best one
    model_trainer = ModelTrainer()

    r2_score = model_trainer.initiate_model_trainer(
        train_array,
        test_array
    )

    print("Training complete.")
    print("R2 score:", r2_score)