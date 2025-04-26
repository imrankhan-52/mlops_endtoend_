import sys
import pandas as pd
import numpy as np
from typing import Optional

from src.configuration.mongodb_connection import MongoDBClient
from src.exception.exception import CustomException
from src.logger.logger import logging

class FetchData:
    """
    A class to export MongoDB records as a pandas DataFrame.
    """

    def __init__(self) -> None:
        """
        Initializes the MongoDB client connection.
        """
        try:
            self.mongo_client = MongoDBClient(database_name='data_gemstone')
        except Exception as e:
            raise CustomException(e, sys)

    def export_collection_as_dataframe(self, collection_name: str, database_name: Optional[str] = None) -> pd.DataFrame:
        """
        Exports an entire MongoDB collection as a pandas DataFrame.

        Parameters:
        ----------
        collection_name : str
            The name of the MongoDB collection to export.
        database_name : Optional[str]
            Name of the database (optional). Defaults to DATABASE_NAME.

        Returns:
        -------
        pd.DataFrame
            DataFrame containing the collection data, with '_id' column removed and 'na' values replaced with NaN.
        """
        try:
            # Access specified collection from the default or specified database
            if database_name is None:
                db=self.mongo_client.database
                collection = db[collection_name]
            else:
                collection = self.mongo_client[database_name][collection_name]


            if collection.count_documents({}) > 0:
                logging.info("Data already exists in MongoDB. Fetching from collection...")
            else:
                logging.info("No data found in MongoDB. Uploading from CSV...")
                df = pd.read_csv("C:\\Users\\HP\\Desktop\\gemstone.csv")
                data_dict = df.to_dict("records")
                collection.insert_many(data_dict)
                logging.info("Data uploaded to MongoDB successfully.")


            # Convert collection data to DataFrame and preprocess
            logging.info("Fetching data from MongoDB")
            docs=collection.find()
            for d in docs:
                print(d)
            df = pd.DataFrame(list(docs))
            logging.info(f"DataFrame shape: {len(df)}")
            return df

        except Exception as e:
            raise CustomException(e,sys)
        


obj = FetchData()
dataframe = obj.export_collection_as_dataframe(collection_name='Cluster0')