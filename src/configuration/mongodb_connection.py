import os
import sys
import pymongo
from pymongo.server_api import ServerApi
from dotenv import load_dotenv
from src.exception.exception import CustomException
from src.logger.logger import logging

load_dotenv()


class MongoDBClient:
    """
    MongoDBClient is responsible for establishing a connection to the MongoDB database

    Attributes:
    client : MongoClient
        A shared MongoClient instance for the class.
    database : Database
        The specific database instance that MongoDBClient connects to.

    Methods:
    -------
    __init__(database_name: str) -> None
        Initializes the MongoDB connection using the given database name.
    """

    client = None  # Shared MongoClient instance across all MongoDBClient instances

    def __init__(self, database_name:str='data_gemstone') -> None:
        """
        Initializes a connection to the MongoDB database. If no existing connection is found, it establishes a new one.

        Parameters:
        ----------
        database_name : str, optional
            Name of the MongoDB database to connect to. Default is set by DATABASE_NAME constant.

        Raises:
        ------
        MyException
            If there is an issue connecting to MongoDB or if the environment variable for the MongoDB URL is not set.
        """
        try:
            # Check if a MongoDB client connection has already been established; if not, create a new one
            if MongoDBClient.client is None:
                mongo_db_url = os.getenv("CONNECTION_URL")  # Retrieve MongoDB URL from environment variables
                if mongo_db_url is None:
                    raise Exception(f"Environment variable  is not set.")
                
                # Establish a new MongoDB client connection
                MongoDBClient.client = pymongo.MongoClient(mongo_db_url,server_api=ServerApi('1'),connectTimeoutMS=300000, socketTimeoutMS=300000)
                
                MongoDBClient.client.admin.command('ping')
                logging.info("Pinged your deployment. You successfully connected to MongoDB!")

                
            # Use the shared MongoClient for this instance
            self.client = MongoDBClient.client
            self.database = self.client[database_name]  # Connect to the specified database
            self.database_name = database_name
            logging.info("Exit Mongodb client.")
            
        except Exception as e:
            # Raise a custom exception with traceback details if connection fails
            raise CustomException(e, sys)


