import os
import sys
from src.logger.logger import logging
from src.exception.exception import CustomException

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

from src.entity.artifacts_entity import (DataIngestionArtifacts,
                                            DataValidationArtifact,
                                            DataTransformationArtifacts,
                                            TrainerModelArtifacts,
                                            RegressionMetricArtifact)



class TraingingPipeline:
    def __init__(self):
        logging.info("Training Pipeline started")

    def start_data_ingestion(self):
        """
        Method Name :   start_data_ingestion
        Description :   This method initiates the data ingestion components of training pipeline

        """
        try:
            data_ingestion = DataIngestion()
            data_ingestion_artifacts = data_ingestion.InitiateDataIngestion()
            return data_ingestion_artifacts
        except Exception as e:
            raise CustomException(e, sys) from e
        

    def start_data_validation(self, data_ingestion_artifacts: DataIngestionArtifacts):
        """
        Method Name :   start_data_validation
        Description :   This method initiates the data validation components of training pipeline

        """
        try:
            data_validation = DataValidation(data_ingestion_artifacts)
            data_validation_artifacts = data_validation.initiate_data_validation()
            return data_validation_artifacts
        except Exception as e:
            raise CustomException(e, sys) from e
        

    def start_data_transformation(self, data_ingestion_artifacts: DataIngestionArtifacts,
                                   data_validation_artifacts: DataValidationArtifact):
        """
        Method Name :   start_data_transformation
        Description :   This method initiates the data transformation components of training pipeline

        """
        try:
            data_transformation = DataTransformation(data_ingestion_artifacts=data_ingestion_artifacts,
                                                     data_validation_artifacts=data_validation_artifacts)
            data_transformation_artifacts=data_transformation.initiate_data_transformation()
            return data_transformation_artifacts
        except Exception as e:
            raise CustomException(e, sys) from e
        

    def start_model_trainer(self, data_transformation_artifacts: DataTransformationArtifacts):
        """
        Method Name :   start_model_trainer
        Description :   This method initiates the model trainer components of training pipeline

        """
        try:
            model_trainer = ModelTrainer(data_transformation_artifacts)
            model_trainer_artifacts = model_trainer.initiate_model_training()
            return model_trainer_artifacts
        except Exception as e:
            raise CustomException(e, sys) from e
        

    def run_pipeline(self):
        """
        Method Name :   run_pipeline
        Description :   This method runs the entire training pipeline

        """
        try:
            data_ingestion_artifacts = self.start_data_ingestion()
            data_validation_artifacts = self.start_data_validation(data_ingestion_artifacts)
            data_transformation_artifacts = self.start_data_transformation(data_ingestion_artifacts,
                                                                           data_validation_artifacts)
            model_trainer_artifacts = self.start_model_trainer(data_transformation_artifacts)
        except Exception as e:
            raise CustomException(e, sys) from e
        

if __name__=="__main__":
    pass
