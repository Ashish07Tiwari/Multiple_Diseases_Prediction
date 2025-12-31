import os
import sys

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from dataclasses import dataclass

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

@dataclass
class ModelTrainerConfig:
    model_dir: str= "Models"

    target_column_map = {
        "breast_cancer": 1,
        "heart": -1,
        "kidney": -1,
        "diabetes": -1
    }

    def get_model_path(self, dataset_name: str) -> str:
        return os.path.join(
            self.model_dir, dataset_name, "model.pkl"
        )
    
    def get_target_index(self, dataset_name: str) -> int:
        if dataset_name not in self.target_column_map:
            raise ValueError(f"Target column index not defined for {dataset_name}")
        return self.target_column_map[dataset_name]

    
class ModelTrainer:
    def __init__(self):
        self.trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, dataset_name, train_arr, test_arr):
        try:
            
            logging.info(f"Model training start for dataset: {dataset_name}")

            target_idx = self.trainer_config.get_target_index(dataset_name)

            if target_idx == 1:
                X_train = train_arr[:, 1:]
                y_train = train_arr[:, 1].astype(int)

                X_test = test_arr[:, 1:]
                y_test = test_arr[:, 0].astype(int)
            else:
                X_train = train_arr[:, :-1]
                y_train = train_arr[:, -1].astype(int)

                X_test = test_arr[:, :-1]
                y_test = test_arr[:, -1].astype(int)

            label_encoder = LabelEncoder()
            y_train = label_encoder.fit_transform(y_train)
            y_test = label_encoder.transform(y_test)

            models = {
                "Logistic Regression": LogisticRegression(),
                "SVM": SVC(),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42),
                "XGBoost": XGBClassifier()
            }

            model_scores = {}

            for model_name,model in models.items():
                logging.info(f"Training model: {model_name}")

                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                score = accuracy_score(y_test, pred)

                model_scores[model_name] = score

            best_model_name = max(model_scores, key=model_scores.get)
            best_model_score = model_scores[best_model_name]
            best_model = models[best_model_name]

            if best_model_score < 0.75:
                raise CustomException("No suitable model found")

            logging.info(
                f"Best model for {dataset_name}: "
                f"{best_model_name} | Score: {best_model_score}"
            )

            model_path = self.trainer_config.get_model_path(dataset_name)
            os.makedirs(os.path.dirname(model_path), exist_ok=True)

            save_object(
                file_path=model_path,
                obj=best_model
            )

            final_preds = best_model.predict(X_test)
            final_score = accuracy_score(y_test, final_preds)

            return {
                "dataset": dataset_name,
                "best_model": best_model_name,
                "score": final_score
            }


        except Exception as e:
            raise CustomException(e, sys)
        