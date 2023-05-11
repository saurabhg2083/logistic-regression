import os
import sys
from src.logger import logging
import numpy as np
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import statsmodels.api as sm
from src.components.data_transformation import DataTransformation
from patsy import dmatrices

## Intitialize the Data Ingetion Configuration

@dataclass
class DataIngestionconfig:
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','raw.csv')

## create a class for Data Ingestion
class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionconfig()

    def initiate_data_ingestion(self):
        logging.info('Data Ingestion methods Starts')
        try:
            #df=pd.read_csv(os.path.join('notebooks/data','gemstone.csv'))
            dta = sm.datasets.fair.load_pandas().data
            dta["affair"] = (dta.affairs > 0).astype(int)
            y,X=dmatrices('affair~rate_marriage + age + yrs_married + children +  religious + educ + C(occupation)+C(occupation_husb)',dta,return_type="dataframe")
            X = X.rename(columns = {'C(occupation)[T.2.0]':'occ_2','C(occupation)[T.3.0]':'occ_3','C(occupation)[T.4.0]':'occ_4','C(occupation)[T.5.0]':'occ_5','C(occupation)[T.6.0]':'occ_6','C(occupation_husb)[T.2.0]':'occ_husb_2','C(occupation_husb)[T.3.0]':'occ_husb_3','C(occupation_husb)[T.4.0]':'occ_husb_4','C(occupation_husb)[T.5.0]':'occ_husb_5','C(occupation_husb)[T.6.0]':'occ_husb_6'})

            df = pd.DataFrame(data = np.c_[X, y], columns=['Intercept','occ_2','occ_3','occ_4','occ_5','occ_6','occ_husb_2','occ_husb_3','occ_husb_4','occ_husb_5','occ_husb_6','rate_marriage','age','yrs_married','children','religious','educ','affair'])

            logging.info('Dataset read as pandas Dataframe')
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False)
            logging.info('Train test split')
            train_set,test_set=train_test_split(df,test_size=0.30,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info('Ingestion of Data is completed')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
  
        except Exception as e:
            logging.info('Exception occured at Data Ingestion stage')
            raise CustomException(e,sys)