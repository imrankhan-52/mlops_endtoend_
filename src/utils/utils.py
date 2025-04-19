import os
import sys
import pickle
import numpy as np
import pandas as pd

from src.logger.logger import logging
from src.exception.exception import CustomException

from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

def save_object(filepath,obj):
    try:
        dir_path=os.path.dirname(filepath)
        os.makedirs(dir_path,exist_ok=True)
        with open(filepath,'wb') as file_obj:
            pickle.dump(obj,file_obj)

    except Exception as e:
        logging.info("Exception occured in save_object function utils.utils")
        raise CustomException(e,sys)


def evaluat_model(X_train,Y_train,X_test,Y_test,models):
    try:
        dict1={}
        for i,model in models.items():
            model.fit(X_train,Y_train)

            y_pred=model.predict(X_test)
            r2=r2_score(Y_test,y_pred)

            dict1[model]=r2
        return max(dict1.items(),key=lambda x:x[1]) 
    except Exception as e:
        raise CustomException(e,sys)


def load_object(file_path):
    try:
        with open(file_path,'rb') as obj:
            return pickle.load(obj)
    except Exception as e:
        logging.info("Exception occured in load_object function utils.utils")
        raise CustomException(e,sys)


def save_numpy_array_data(file_path: str, array: np.array):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise CustomException(e, sys) from e