import os
from pathlib import Path
from dataclasses import dataclass

@dataclass
class DataIngestionArtifacts:
    """
    DataIngestionArtifacts:

    train_data_path : after splitting data train data will be store
    test_data_path : after splitting test data will be store
    
    """
    train_data_path:Path
    test_data_path:Path



@dataclass
class DataValidationArtifact:
    validation_status:bool
    message: str
    validation_report_file_path: str


@dataclass
class DataTransformationArtifacts:
    train_arr_path:Path
    test_arr_path:Path
    preprocessor_obj_file_path:Path


@dataclass
class RegressionMetricArtifact:
    r2_score:float
    mse:float
    mae:float


@dataclass
class TrainerModelArtifacts:
    trained_model_file_path:Path
    metric_artifact: RegressionMetricArtifact

