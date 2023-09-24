import logging
import pandas as pd
from src.data_cleaning import DataCleaning, DataPreprocessStrategy

def get_data_for_test():
    try:
      df = pd.read_csv("/home/lohithun97/cy_prediction/data/yield_df.csv")
      process_strategy = DataPreprocessStrategy()
      data_cleaning = DataCleaning(df, process_strategy)
      df = data_cleaning.handle_data()
      df.drop(['hg/ha_yield'], axis=1, inplace=True)
      df = df.sample(100)
      result = df.to_json(orient="split")
      return result
    except Exception as e:
      logging.error(e)
      raise e
      