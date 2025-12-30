import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder 

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformerConfig:
    preprocessor_obj_file_path: str

DATASET_CONFIGS = {

    "breast_cancer": {
        "numerical_columns": [
            "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
            "smoothness_mean", "compactness_mean", "concavity_mean",
            "concave points_mean", "symmetry_mean", "fractal_dimension_mean"
        ],
        "categorical_columns": [],
        "target_column": "diagnosis"
    },

    "heart": {
        "numerical_columns": [
            "age", "trestbps", "chol", "thalach", "oldpeak"
        ],
        "categorical_columns": [
            "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"
        ],
        "target_column": "target"
    },

    "kidney": {
        "numerical_columns": [
            "age", "bp", "bgr", "bu", "sc", "sod", "pot",
            "hemo", "pcv", "wc", "rc", "sg", "al", "su"
        ],
        "categorical_columns": [
            "rbc", "pc", "pcc", "ba", "htn", "dm",
            "cad", "appet", "pe", "ane"
        ],
        "target_column": "classification"
    },

    "diabetes": {
        "numerical_columns": [
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age"
        ],
        "categorical_columns": [],
        "target_column": "Outcome"
    }
}

class DataTransformer:
    def __init__(self, dataset_name: str):
        if dataset_name not in DATASET_CONFIGS:
            raise ValueError(f"Dataset config not found for: {dataset_name}")
        
        config = DATASET_CONFIGS[dataset_name]

        self.numerical_features = config["numerical_columns"]
        self.categorical_features = config["categorical_columns"]
        self.target_feature = config["target_column"]

        self.transformer_config = DataTransformerConfig(
            preprocessor_obj_file_path= os.path.join('artifacts', dataset_name, "preprocessor.pkl")
        )

    def get_data_transformer_object(self):
        try:
            
            numerical_pipeline= Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy= "median")),
                    ("scaler", RobustScaler())
                ]
            )

            categorical_pipeline= Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ("scaler", RobustScaler(with_centering=False))
                ]
            )

            logging.info(f"Numerical Columns: {self.numerical_features}")
            logging.info(f"Categorical Columns: {self.categorical_features}")

            preprocessor= ColumnTransformer(
                transformers=[
                    ("numerical_pipeline", numerical_pipeline, self.numerical_features),
                    ("categorical_pipeline", categorical_pipeline, self.categorical_features)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformer(self, train_path: str, test_path: str):
        try:
            
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            train_data.replace(['?', '\t?'], np.nan, inplace=True)
            test_data.replace(['?', '\t?'], np.nan, inplace=True)


            logging.info("Train and Test data loaded")

            X_train = train_data.drop(columns=[self.target_feature])
            y_train = train_data[self.target_feature]

            X_test = test_data.drop(columns= [self.target_feature])
            y_test = test_data[self.target_feature]

            preprocessor_obj = self.get_data_transformer_object()

            X_train_arr = preprocessor_obj.fit_transform(X_train)
            X_test_arr = preprocessor_obj.transform(X_test)

            train_arr = np.c_[X_train_arr, y_train.to_numpy()]
            test_arr = np.c_[X_test_arr, y_test.to_numpy()]

            os.makedirs(os.path.dirname(self.transformer_config.preprocessor_obj_file_path), exist_ok= True)

            save_object(
                file_path=self.transformer_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            logging.info("Preprocessing completed with RobustScaler")

            return train_arr, test_arr, self.transformer_config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)
