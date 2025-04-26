import os
import sys
import mlflow
import numpy as np
import pandas as pd

from pathlib import Path
from dataclasses import dataclass
from src.logger.logger import logging
from src.exception.exception import CustomException
from src.utils.utils import save_object,save_numpy_array_data
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation

from src.entity.artifacts_entity import DataIngestionArtifacts,DataValidationArtifact,DataTransformationArtifacts
from src.entity.config_entity import DataTransformationConfig

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder,StandardScaler
from sklearn.impute import SimpleImputer



   
class DataTransformation:
    def __init__(self,data_ingestion_artifacts:DataIngestionArtifacts,
                 data_validation_artifacts:DataValidationArtifact):

        self.data_transformation_config=DataTransformationConfig()
        self.data_ingestion_artifacts=data_ingestion_artifacts
        self.data_validation_artifacts=data_validation_artifacts

    def data_transformation(self,gms):
        try:
            logging.info("Data Transformation started")
            cut_map=['Fair','Good','Very Good','Premium','Ideal']
            clarity_map=['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF'] 
            color_map=["D",'E','F','G','H','I','J']

            cat_col=gms.select_dtypes(include='object').columns
            num_col=gms.select_dtypes(exclude='object').columns

            num_col=num_col.drop(['price'])

            num_pipeline=Pipeline(steps=[
                ('impute',SimpleImputer()),
                ('scaler',StandardScaler())
            ])


            cat_pipeline=Pipeline(steps=[
                ('imputer2',SimpleImputer(strategy='most_frequent')),
                ('ordinalencoder',OrdinalEncoder(categories=[cut_map,color_map,clarity_map]))
            ])

            ct=ColumnTransformer(
            [
                ('num_pipeline',num_pipeline,num_col),
                ('cat_pipeline',cat_pipeline,cat_col)
                ],remainder='passthrough'
            )
            logging.info("Data Transformation object is returned")
            return ct
        except Exception as e:
            logging.info("data")
            raise CustomException(e,sys)

    def initiate_data_transformation(self)->DataTransformationArtifacts:
        """
        Initiating Data Transformation Artifacts 
        """
        try:
            mlflow.set_experiment("GemStone_ML_Pipeline")
            with mlflow.start_run(run_name="data_transformation",nested=True):
                if not self.data_validation_artifacts.validation_status:
                    raise Exception(self.data_validation_artifacts.message)
                train=pd.read_csv(self.data_ingestion_artifacts.train_data_path)
                test=pd.read_csv(self.data_ingestion_artifacts.test_data_path)

                logging.info(f"train DataFrame Head : \n{train.head()}")
                logging.info(f"test DataFrame Head : \n{test.head()}")
                
                preprocessor_obj=self.data_transformation(train)

                target_column='price'
                drop_column=[target_column]

                train_df = train.drop(drop_column,axis=1)
                target_train=train[target_column]

                test_df = test.drop(drop_column,axis=1)
                target_test=test[target_column]

                logging.info("Applying preprocessing object on training and testing datasets \n")

                train_arr=preprocessor_obj.fit_transform(train_df)
                test_arr=preprocessor_obj.transform(test_df)


                train_arr=np.c_[train_arr,np.array(target_train)]
                test_arr=np.c_[test_arr,np.array(target_test)]

                save_numpy_array_data(self.data_transformation_config.train_arr_path,train_arr)
                save_numpy_array_data(self.data_transformation_config.test_arr_path,test_arr)


                #mlflow.log_artifact(self.data_transformation_config.preprocessor_obj_file_path)

                mlflow.log_param("transformed_train_shape", train_arr.shape)
                mlflow.log_param("transformed_test_shape", test_arr.shape)

                save_object(self.data_transformation_config.preprocessor_obj_file_path,preprocessor_obj)
                logging.info("preprocessor object pickle file is saved \n")
                logging.info("Data preprocessing step is completed \n")
                return DataTransformationArtifacts(
                    preprocessor_obj_file_path=self.data_transformation_config.preprocessor_obj_file_path,
                    train_arr_path=self.data_transformation_config.train_arr_path,
                    test_arr_path=self.data_transformation_config.test_arr_path
                    )            
        except Exception as e:
            raise CustomException(e,sys)
        

if __name__=="__main__":
    pass