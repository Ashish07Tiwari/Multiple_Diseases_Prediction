import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from dataclasses import dataclass
from sklearn.model_selection import train_test_split

@dataclass
class DataIngestionConfig:
    artifactes_dir: str= "artifacts"

class DataIngestion:
    def __init__(self, dataset_path: str, dataset_name: str):
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info(f"Data Ingestion started for {self.dataset_name}")

        try:
            dataset = pd.read_csv(self.dataset_path)

            dataset_artifact_dir = os.path.join(
                self.ingestion_config.artifactes_dir,
                self.dataset_name
            )

            os.makedirs(dataset_artifact_dir, exist_ok=True)

            raw_path = os.path.join(dataset_artifact_dir, "data.csv")
            train_path = os.path.join(dataset_artifact_dir, "train.csv")
            test_path = os.path.join(dataset_artifact_dir, "test.csv")

            dataset.to_csv(raw_path, index=False, header=True)

            train_set, test_set = train_test_split(dataset, test_size= 0.2, random_state= 42)

            train_set.to_csv(train_path, index= False, header= True)
            test_set.to_csv(test_path, index= False, header= True)

            logging.info(f"Ingestion completed for {self.dataset_name}")

            return train_path, test_path

        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    # For Breast-Cancer dataset
    ingestion = DataIngestion(r"notebooks\data\breast-cancer.csv", "Breast-Cancer")
    train_path_bc, test_path_bc = ingestion.initiate_data_ingestion()

    # For Heart Diseases dataset
    ingestion = DataIngestion(r"notebooks\data\heart.csv", "Heart-Diseases")
    train_path_heart, test_path_heart = ingestion.initiate_data_ingestion()

    # For Diabetes Diseases dataset
    ingestion = DataIngestion(r"notebooks\data\diabetes.csv", "Diabetes-Diseases")
    train_path_diabetes, test_path_diabetes = ingestion.initiate_data_ingestion()

    # For Kidney Diseases dataset
    ingestion = DataIngestion(r"notebooks\data\kidney_disease.csv", "Kidney-Diseases")
    train_path_kidney, test_path_kidney = ingestion.initiate_data_ingestion()
