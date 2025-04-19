import os
import sys
from src.logger.logger import logging
from src.exception.exception import CustomException

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation

try:
    logging.info("Training Pipeline::::::::")
    logging.info("Data ingestion is started::::::")
    obj=DataIngestion()
    train,test=obj.InitiateDataIngestion()
    logging.info("Data ingested:::::::")

    logging.info("Data transformatin is started::::::")
    obj2=DataTransformation()
    train_arr,test_arr=obj2.initiate_data_transformation(train,test)
    logging.info("Data has Transformed")

    logging.info("Model Trainer is started::::::")
    obj3=ModelTrainer()
    obj3.initiate_model_training(train_arr,test_arr)
    logging.info("Model has Trained Succesfully:::::::")

    logging.info("Model evaluation:::::::::")
    obj4=ModelEvaluation()
    obj4.initiate_model_evaluation(test_arr)
    logging.info("Model Evaluation completed:::::::::")

except Exception as e:
    raise CustomException(e,sys)