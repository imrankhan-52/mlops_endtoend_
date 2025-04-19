import os
import sys
import mlflow
import mlflow.artifacts
from mlflow.sklearn import autolog

import numpy as np
import pandas as pd

from dataclasses import dataclass
from pathlib import Path

from src.logger.logger import logging
from src.exception.exception import CustomException
from src.utils.utils import save_object,evaluat_model
from sklearn.linear_model import LinearRegression,Lasso

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation


@dataclass
class ModelTrainerConfig:
    trained_model_path=os.path.join("model","model.pkl")
    

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_training(self,train_arr,test_arr):
        try:
            mlflow.set_experiment("GemStone_ML_Pipeline")
            autolog()
            with mlflow.start_run(run_name="model_trainer"):
                logging.info("Splitting Dependent and independent variable")
                X_train=train_arr[:,:-1]
                logging.info(f"X_train shape  : {pd.DataFrame(X_train).shape} \n  {pd.DataFrame(X_train).head(2)}\n")
                
                Y_train=train_arr[:,-1]
                logging.info(f"Y_train shape  : {pd.DataFrame(Y_train).shape} \n  {pd.DataFrame(Y_train).head(2)}\n")
                
                X_test=test_arr[:,:-1]
                logging.info(f"X_test shape  : {pd.DataFrame(X_test).shape} \n  {pd.DataFrame(X_test).head(2)}\n")
                
                Y_test=test_arr[:,-1]
                logging.info(f"Y_test shape  : {pd.DataFrame(Y_test).shape} \n  {pd.DataFrame(Y_test).head(2)}\n")

                dict1={
                'LinearRegression':LinearRegression(),
                'Lasso':Lasso(),
                #'RandomForestRegressor':RandomForestRegressor(),
                #'GradientBoostingRegressor':GradientBoostingRegressor()
                }

                logging.info("model evaluation :")
                
                best_model=evaluat_model(X_train,Y_train,X_test,Y_test,dict1)

                logging.info(f"Best model is ::: {best_model} \n")
                model_obj=best_model[0]
                logging.info(f"Model is Training is completed")
                logging.info(f"Model path is logged to mlflow")
                mlflow.log_artifact(self.model_trainer_config.trained_model_path)
                save_object(self.model_trainer_config.trained_model_path,model_obj)


        except Exception as e:
            logging.info("model trainer components error")
            raise CustomException(e,sys)
        

