import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model_path=os.path.join('artifacts','model.pkl')

            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)

            data_scaled=preprocessor.transform(features)

            pred=model.predict(data_scaled)
            return pred
            

        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(self,
                    Intercept:float,
                    occ_2:float,
                    occ_3:float,
                    occ_4:float,
                    occ_5:float,
                    occ_6:float,
                    occ_husb_2:float,
                    occ_husb_3:float,
                    occ_husb_4:float,
                    occ_husb_5:float,
                    occ_husb_6:float,
                    rate_marriage:float,
                    age:float,
                    yrs_married:float,
                    children:float,
                    religious:float,
                    educ:float):
        
        self.Intercept=Intercept
        self.occ_2=occ_2
        self.occ_3=occ_3
        self.occ_4=occ_4
        self.occ_5=occ_5
        self.occ_6=occ_6
        self.occ_husb_2 = occ_husb_2
        self.occ_husb_3 = occ_husb_3
        self.occ_husb_4=occ_husb_4
        self.occ_husb_5=occ_husb_5
        self.occ_husb_6 = occ_husb_6
        self.rate_marriage = rate_marriage
        self.age=age
        self.yrs_married=yrs_married
        self.children = children
        self.religious = religious
        self.educ=educ

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'Intercept':[self.Intercept],
                'occ_2':[self.occ_2],
                'occ_3':[self.occ_3],
                'occ_4':[self.occ_4],
                'occ_5':[self.occ_5],
                'occ_6':[self.occ_6],
                'occ_husb_2':[self.occ_husb_2],
                'occ_husb_3':[self.occ_husb_3],
                'occ_husb_4':[self.occ_husb_4],
                'occ_husb_5':[self.occ_husb_5],
                'occ_husb_6':[self.occ_husb_6],
                'rate_marriage':[self.rate_marriage],
                'age':[self.age],
                'yrs_married':[self.yrs_married],
                'children':[self.children],
                'religious':[self.religious],
                'educ':[self.educ]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e,sys)
