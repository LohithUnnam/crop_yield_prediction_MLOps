import logging
from abc import ABC, abstractmethod
from typing import Union
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataStrategy(ABC):
      """
      Abstract class defining strategy for handling data
      """

      @abstractmethod
      def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
            pass


class DataPreprocessStrategy(DataStrategy):
      """
      Strategy for cleaning data
      """
      
      def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
           """
           cleaning the data
           """ 
           try:
              data = data.drop(columns=['Unnamed: 0'])
              num_cols = list(data.select_dtypes(exclude=object).columns) # list of numerical columns
              num_cols.remove("Year")
              num_cols.remove('hg/ha_yield')
              cat_cols = list(data.select_dtypes(include=object).columns) # list of categorical columns
              cat_cols.append("Year")
              data = pd.get_dummies(data, columns=cat_cols)
              data.replace({True:1, False:0}, inplace=True)
              sc = StandardScaler()
              data[num_cols] = sc.fit_transform(data[num_cols])
              return data              
           except Exception as e:
               logging.error(f"Error in cleaning data: {e}")
               raise e   


class DataDivideStrategy(DataStrategy):
     """
     Strategy for dividing the data into train and test sets
     """   

     def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
          """
          Divides the data into train and test
          """
          try:
            y = data['hg/ha_yield']
            X = data.drop('hg/ha_yield',axis=1)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=7)
            return X_train, X_test, y_train, y_test
          except Exception as e:
            logging.error(f"Error in dividing data: {e}")
            raise e
          
class DataCleaning:
      """
      This class is used to do all the preprocessing of data and then dividing it into training and testing sets.
      """
      def __init__(self, data: pd.DataFrame, strategy: DataStrategy) -> None:
          self.data = data
          self.strategy = strategy

      def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
           """
           Handle data
           """    
           try:
              return self.strategy.handle_data(self.data)
           except Exception as e:
              logging.error(f"Error in handling data: {e}")
              raise e