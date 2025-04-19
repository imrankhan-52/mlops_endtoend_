import os
import sys
import pandas as pd

from src.logger.logger import logging
from src.exception.exception import CustomException
from src.utils.utils import load_object


class PredictionPipeline:
    def __init__(self):
        print("init .... the object")

    def predict(self,feature):
        try:
            preprocessor_path=os.path.join("data","processed","preprocessor.pkl")
            model_path=os.path.join("model","model.pkl")

            preprocessor_obj=load_object(preprocessor_path)
            model_obj=load_object(model_path)

            scaled_feature=preprocessor_obj.transform(feature)
            pred=model_obj.predict(scaled_feature)
            return pred

        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(self,carat:float,depth:float,table:float,x:float,y:float,z:float,cut:str,color:str,clarity:str):
        self.carat=carat
        self.depth=depth
        self.table=table
        self.x=x
        self.y=y
        self.z=z
        self.cut=cut
        self.color=color
        self.clarity=clarity
        
    def get_data_as_dataframe(self):
        try:
            dict1={
            'carat':[self.carat],
            'depth':[self.depth],
            'table':[self.table],
            'x':[self.x],
            'y':[self.y],
            'z':[self.z],
            'cut':[self.cut],
            'color':[self.color],
            'clarity':[self.clarity]
        }
            df=pd.DataFrame(dict1)
            logging.info(df)
            logging.info("DataFrame Gathered::::")
            return df
            
        except Exception as e:
            logging.info('error while creating dataframe')
            raise CustomException(e,sys)
