import os
from dataclasses import dataclass
from pathlib import Path



@dataclass
class DataIngestionConfig:
    """
    DataIngestionConfig:

    raw_data_path : store raw data 
    train_data_path : after splitting data train data will be store
    test_data_path : after splitting test data will be store

    """
    raw_data_path:Path=os.path.join("data","raw","raw.csv")
    train_data_path:Path=os.path.join("data","raw","train.csv")
    test_data_path:Path=os.path.join("data","raw","test.csv")
    params_yaml_path:Path=Path("params.yaml")
    collection_name:str=""


@dataclass
class DataValidationConfig:
    validation_report_file_path: str = os.path.join("report","report.yaml")



@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join("data","processed","preprocessor.pkl")
    train_arr_path=os.path.join("data","process","train_arr.np")
    test_arr_path=os.path.join("data","process","test_arr.np")


@dataclass
class TrainerModelConfig:
    trained_model_path=os.path.join("model","model.pkl")

