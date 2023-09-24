import logging
from abc import abstractmethod, ABC
from sklearn.tree import DecisionTreeRegressor

class Model(ABC):
      """
      Abstract class for all models
      """

      @abstractmethod
      def train(self, X_train, y_train):
            """
            Trains the model

            Args:
              X_train: Training data
              y_test: Training labels

            Returns:
               None
            """
            pass


class DTRegressor(Model):
      """
      Decision Tree Regressor Model
      """

      def train(self, X_train, y_train, **kwargs):
            """
            Trains the model

            Args:
              X_train: Training data
              y_train: Testing data

            Returns:
              None
            """
            try:
              reg = DecisionTreeRegressor(**kwargs)
              reg.fit(X_train, y_train)
              logging.info("Model training completed")
              return reg
            except Exception as e:
                 logging.error(f"Error while training the model: {e}")
                 raise e