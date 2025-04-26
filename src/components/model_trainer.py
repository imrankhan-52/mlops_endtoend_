import os
import sys
from cloudpickle import load
from typing import Tuple
import mlflow
import mlflow.artifacts
from mlflow.sklearn import autolog

import numpy as np
import pandas as pd

from dataclasses import dataclass
from pathlib import Path

from src.entity.artifacts_entity import (DataTransformationArtifacts,
                                        TrainerModelArtifacts,
                                        RegressionMetricArtifact)
from src.entity.config_entity import TrainerModelConfig
from src.logger.logger import logging
from src.exception.exception import CustomException

from sklearn.linear_model import LinearRegression,Lasso
from src.utils.utils import load_object,load_numpy_array_data,save_object
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation


class ModelTrainer:
    def __init__(self,data_transformation_artifacts:DataTransformationArtifacts):

        self.model_trainer_config=TrainerModelConfig()
        self.data_transformation_artifacts=data_transformation_artifacts

    def get_model_object_and_report(self, train: np.array, test: np.array) -> Tuple[object, object]:
        """
        Method Name :   get_model_object_and_report
        Description :   This function trains a RandomForestClassifier with specified parameters
        
        Output      :   Returns metric artifact object and trained model object
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            logging.info("Training LassoRegression with specified parameters")

            # Splitting the train and test data into features and target variables
            x_train, y_train, x_test, y_test = train[:, :-1], train[:, -1], test[:, :-1], test[:, -1]
            logging.info("train-test split done.")

            model=Lasso(alpha=0.1, max_iter=100, random_state=42)

            # Fit the model
            logging.info("Model training going on...")
            model.fit(x_train, y_train)
            logging.info("Model training done.")

            # Predictions and evaluation metrics
            y_pred = model.predict(x_test)
            accuracy = r2_score(y_test, y_pred)
            mae=mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            #rmse=np.sqrt(mean_squared_error((y_test, y_pred)))
            metric_artifact=RegressionMetricArtifact(r2_score=accuracy,mae=mae,mse=mse)

            return model, metric_artifact
        
        except Exception as e:
            raise CustomException(e, sys) from e



    def initiate_model_training(self):
        """
        Method Name :   initiate_model_trainer
        Description :   This function initiates the model training steps
        
        Output      :   Returns model trainer artifact
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            mlflow.set_experiment("GemStone_ML_Pipeline")
            autolog()
            with mlflow.start_run(run_name="model_trainer"):
                logging.info("Model training starting:")
                train_arr=load_numpy_array_data(self.data_transformation_artifacts.train_arr_path)
                test_arr=load_numpy_array_data(self.data_transformation_artifacts.test_arr_path)
                logging.info(f"train_arr shape  : {pd.DataFrame(train_arr).shape} \n  {pd.DataFrame(train_arr).head(2)}\n")

                trained_model, metric_artifact = self.get_model_object_and_report(train=train_arr, test=test_arr)
                logging.info(f"model metrics {metric_artifact}")
                logging.info("Model object and artifact loaded.")
                save_object(self.model_trainer_config.trained_model_path,trained_model)

                # Load preprocessing object
                preprocessing_obj = load_object(file_path=self.data_transformation_artifacts.preprocessor_obj_file_path)
                logging.info("Preprocessing obj loaded.")

                if r2_score(train_arr[:, -1], trained_model.predict(train_arr[:, :-1])) < 0.80:
                    logging.info("No model found with score above the base score")
                    raise Exception("No model found with score above the base score")

                model_trainer_artifact = TrainerModelArtifacts(
                trained_model_file_path=self.model_trainer_config.trained_model_path,
                metric_artifact=metric_artifact,
                )
                logging.info("Model Training completed:")
                return model_trainer_artifact

        except Exception as e:
            logging.info("model trainer components error")
            raise CustomException(e,sys)
        

if __name__=="__main__":
    pass