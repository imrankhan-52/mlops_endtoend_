from operator import index
import os
import sys
import mlflow
import numpy as np
import pandas as pd
from pathlib import Path

from src.entity.config_entity import DataIngestionConfig
from src.entity.artifacts_entity import DataIngestionArtifacts
from src.logger.logger import logging
from src.exception.exception import CustomException
from src.utils.utils import read_yaml_file
from sklearn.model_selection import train_test_split

class DataIngestion:
    def __init__(self):
        """
        DataIngestionConfig : for Data Ingestion
        """
        
        self.data_ingestion_config=DataIngestionConfig()


    def export_data_into_feature_store(self):
        """
        Method Name :   export_data_into_feature_store
        Description :   This method exports data from mongodb to csv file
        
        Output      :   data is returned as artifact of data ingestion components
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            logging.info(f"Exporting data from mongodb")
            #obj = FetchData()
            #dataframe = obj.export_collection_as_dataframe(collection_name='Cluster0')
            #logging.info(f"Shape of dataframe: {dataframe.shape}")
            raw_data_path  = self.data_ingestion_config.raw_data_path
            dir_path = os.path.dirname(raw_data_path)
            os.makedirs(dir_path,exist_ok=True)
            logging.info(f"Saving exported data into feature store file path: {raw_data_path}")
            dataframe= pd.read_csv("C:\\Users\\HP\\Desktop\\gemstone.csv")
            dataframe.drop('id',inplace=True,axis=1)
            dataframe.to_csv(raw_data_path,header=True,index=False)
            return dataframe

        except Exception as e:
            raise CustomException(e,sys)



    def split_data_as_train_test(self,dataframe) ->None:
        """
        Method Name :   split_data_as_train_test
        Description :   This method splits the dataframe into train set and test set based on split ratio 
        
        Output      :   Folder is created in s3 bucket
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Entered split_data_as_train_test method of Data_Ingestion class")

        try:
            logging.info("Train Test splite on DataFrame")
            ingestion=read_yaml_file(self.data_ingestion_config.params_yaml_path)
            test_size=ingestion['data_ingestion']['test_size']
            random_state=ingestion['data_ingestion']['random_state']
            logging.info(f"Test size is {test_size}")
            logging.info(f"Random state is {random_state}")

            train_data, test_data = train_test_split(dataframe,test_size=test_size,random_state=42) 
            dir_path = os.path.dirname(self.data_ingestion_config.train_data_path)
            os.makedirs(dir_path,exist_ok=True)
            
            logging.info(f"Exporting train and test file path.")
            train_data.to_csv(self.data_ingestion_config.train_data_path,header=True,index=False)
            test_data.to_csv(self.data_ingestion_config.test_data_path,header=True,index=False)

            logging.info(f"Exported train and test file path.")
        except Exception as e:
            raise CustomException(e, sys)


    def InitiateDataIngestion(self):
        """
        Method Name :   initiate_data_ingestion
        Description :   This method initiates the data ingestion components of training pipeline 
        
        Output      :   train set and test set are returned as the artifacts of data ingestion components
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("data ingestion started:")
        try:
            mlflow.set_experiment("GemStone_ML_Pipeline")
            with mlflow.start_run(run_name="data_ingestion"):


                dataframe = self.export_data_into_feature_store()

                mlflow.log_param("raw_shape", dataframe.shape)
                mlflow.log_param("raw_columns", list(dataframe.columns))

                logging.info("Data ingested successfully:")

                logging.info("DataFrame Created from mongodb ingested data:")

                self.split_data_as_train_test(dataframe)

                #mlflow.log_param('test_size',self.data_ingestion_config.test_size)

                logging.info("Train Test is performed on dataset")

                mlflow.log_param("raw_data_path", self.data_ingestion_config.raw_data_path)
                mlflow.log_param("train_data_path", self.data_ingestion_config.train_data_path)
                mlflow.log_param("test_data_path", self.data_ingestion_config.test_data_path)

                data_ingestion_artifacts=DataIngestionArtifacts(train_data_path=self.data_ingestion_config.train_data_path,
                test_data_path=self.data_ingestion_config.test_data_path)

                logging.info(f"Data ingestin Artifacts {data_ingestion_artifacts}")

                return data_ingestion_artifacts
                
        except Exception as e:
            logging.info("error occured:")
            raise CustomException(e,sys)


if __name__=="__main__":
    obj=DataIngestion()
    obj.InitiateDataIngestion()
