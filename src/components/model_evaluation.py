import os
import sys
import mlflow
from mlflow.sklearn import load_model
from mlflow.models import infer_signature
import numpy as np
import pickle

from src.exception.exception import CustomException
from src.logger.logger import logging
from src.utils.utils import load_object
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


class ModelEvaluation:
    def __init__(self):
        logging.info("evaluation started")
    
    def eval_metrics(self,actual,pred):
        rmse=np.sqrt(mean_squared_error(actual,pred))
        mae=mean_absolute_error(actual,pred)
        r2=r2_score(actual,pred)
        return rmse,mae,r2
    
    def initiate_model_evaluation(self,test_arr):
        try:
            mlflow.set_experiment("GemStone_ML_Pipeline")
            with mlflow.start_run(run_name="model_evaluation",nested=True) as run:
                X_test,y_test=(test_arr[:,:-1],test_arr[:,-1])
                model_path=os.path.join("model","model.pkl")
                model=load_object(model_path)
                prediction=model.predict(X_test)

                rmse,mae,r2=self.eval_metrics(y_test,prediction)
                mlflow.log_metric("rmse",rmse)
                mlflow.log_metric("r2",r2)
                mlflow.log_metric("mae",mae)
                #model_uri = f"runs:/{run.info.run_id}/model"
                #mlflow.register_model(model_uri, "linear_regression")
                #mlflow.register_model(model,"linear_Regression")
                signature=infer_signature(X_test,prediction)
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="sklearn-model",
                    signature=signature,
                    registered_model_name="sk-learn-random-forest-reg-model",
                    )


        except Exception as e:
            raise CustomException(e,sys)


