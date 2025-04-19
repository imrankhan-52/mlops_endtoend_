import os
import sys
import mlflow
import numpy as np
import pandas as pd

from pathlib import Path
from dataclasses import dataclass
from src.logger.logger import logging
from src.exception.exception import CustomException
from sklearn.model_selection import train_test_split


@dataclass
class DataIngestionConfig:
    # first initialize this config
    raw_data_path:str=os.path.join("data","raw.csv")
    train_data_path:str=os.path.join("data","train.csv")
    test_data_path:str=os.path.join("data","test.csv")

class DataIngestion:
    def __init__(self):
        # secound create object of the class
        self.ingestion_config=DataIngestionConfig()

    def InitiateDataIngestion(self):
        logging.info("data ingestion started:")
        try:
            mlflow.set_experiment("GemStone_ML_Pipeline")
            with mlflow.start_run(run_name="data_ingestion"):
                # third reading data
                raw_data_path="data/raw/gemstone.csv" 
                df=pd.read_csv(raw_data_path)
                logging.info("reading data from dataframe")
                mlflow.log_param("raw_data_path",raw_data_path)
                mlflow.log_param("raw_data_shape",df.shape)

                # make directory 
                os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
                # save data to this directory
                df.to_csv(self.ingestion_config.raw_data_path,index=False)
                logging.info("i am performing train test split")

                train_data,test_data=train_test_split(df,test_size=0.25)
                logging.info("saving train and test data .")
                train_data.to_csv(self.ingestion_config.train_data_path,index=False)
                test_data.to_csv(self.ingestion_config.test_data_path,index=False)

                logging.info("data ingestion part completed:")

                return(
                    self.ingestion_config.train_data_path,
                    self.ingestion_config.test_data_path
                )
                
        except Exception as e:
            logging.info("error occured:")
            raise CustomException(e,sys)


