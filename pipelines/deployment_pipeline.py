import numpy as np
import pandas as pd
import json
from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.steps import BaseParameters, Output

from steps.clean_data import clean_df
from steps.evaluation import evaluate_model
from steps.ingest_data import ingest_df
from steps.model_train import train_model
from .utils import get_data_for_test

docker_settings = DockerSettings(required_integrations=[MLFLOW])

class DeploymentTriggerConfig(BaseParameters):
      """Deployment trigger config"""
      min_accuracy = 0.92

@step(enable_cache=False)
def dynamic_importer() -> str:
    data = get_data_for_test()
    return data

@step(enable_cache=False)
def deployment_trigger(
      accuracy: float,
      config: DeploymentTriggerConfig,
):
      """Implements a basic deployment trigger that looks at the input model accuracy and take decison whether to deploy or not"""
      return accuracy > config.min_accuracy

class MLFlowDeploymentLoaderStepParameters(BaseParameters):
    """MLflow deployment getter parameters

    Attributes:
        pipeline_name: name of the pipeline that deployed the MLflow prediction
            server
        step_name: the name of the step that deployed the MLflow prediction
            server
        running: when this flag is set, the step only returns a running service
        model_name: the name of the model that is deployed
    """

    pipeline_name: str
    step_name: str
    running: bool = True

@step(enable_cache=False)
def prediction_service_loader(
      pipeline_name: str,
      pipeline_step_name: str,
      running: bool = True,
      model_name: str = "model",
) -> MLFlowDeploymentService:
    """Get the prediction service started by the deployment pipeline.

    Args:
        pipeline_name: name of the pipeline that deployed the MLflow prediction
            server
        step_name: the name of the step that deployed the MLflow prediction
            server
        running: when this flag is set, the step only returns a running service
        model_name: the name of the model that is deployed
    """
    # get the MLflow model deployer stack component
    mlflow_model_deployer_component = MLFlowModelDeployer.get_active_model_deployer()
    # fetch existing services with same pipeline name, step name and model name
    existing_services = mlflow_model_deployer_component.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        model_name=model_name,
        running=running,
    )

    if not existing_services:
        raise RuntimeError(
            f"No MLflow prediction service deployed by the "
            f"{pipeline_step_name} step in the {pipeline_name} "
            f"pipeline for the '{model_name}' model is currently "
            f"running."
        )
    return existing_services[0]

@step(enable_cache=False)
def predictor(
     service: MLFlowDeploymentService,
     data: str,
) -> np.ndarray :
   service.start(timeout=10)
   data = json.loads(data)
   data.pop("columns")
   data.pop("index")
   
   columns_for_df = ['average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp', 'Area_Albania', 'Area_Algeria', 'Area_Angola', 'Area_Argentina', 'Area_Armenia', 
    'Area_Australia', 'Area_Austria', 'Area_Azerbaijan', 'Area_Bahamas', 'Area_Bahrain', 'Area_Bangladesh', 'Area_Belarus', 'Area_Belgium', 'Area_Botswana', 'Area_Brazil', 
    'Area_Bulgaria', 'Area_Burkina Faso', 'Area_Burundi', 'Area_Cameroon', 'Area_Canada', 'Area_Central African Republic', 'Area_Chile', 'Area_Colombia', 'Area_Croatia', 
    'Area_Denmark', 'Area_Dominican Republic', 'Area_Ecuador', 'Area_Egypt', 'Area_El Salvador', 'Area_Eritrea', 'Area_Estonia', 'Area_Finland', 'Area_France', 
    'Area_Germany', 'Area_Ghana', 'Area_Greece', 'Area_Guatemala', 'Area_Guinea', 'Area_Guyana', 'Area_Haiti', 'Area_Honduras', 'Area_Hungary', 'Area_India', 
    'Area_Indonesia', 'Area_Iraq', 'Area_Ireland', 'Area_Italy', 'Area_Jamaica', 'Area_Japan', 'Area_Kazakhstan', 'Area_Kenya', 'Area_Latvia', 'Area_Lebanon', 
    'Area_Lesotho', 'Area_Libya', 'Area_Lithuania', 'Area_Madagascar', 'Area_Malawi', 'Area_Malaysia', 'Area_Mali', 'Area_Mauritania', 'Area_Mauritius', 'Area_Mexico', 
    'Area_Montenegro', 'Area_Morocco', 'Area_Mozambique', 'Area_Namibia', 'Area_Nepal', 'Area_Netherlands', 'Area_New Zealand', 'Area_Nicaragua', 'Area_Niger', 
    'Area_Norway', 'Area_Pakistan', 'Area_Papua New Guinea', 'Area_Peru', 'Area_Poland', 'Area_Portugal', 'Area_Qatar', 'Area_Romania', 'Area_Rwanda', 'Area_Saudi Arabia',
    'Area_Senegal', 'Area_Slovenia', 'Area_South Africa', 'Area_Spain', 'Area_Sri Lanka', 'Area_Sudan', 'Area_Suriname', 'Area_Sweden', 'Area_Switzerland', 
    'Area_Tajikistan', 'Area_Thailand', 'Area_Tunisia', 'Area_Turkey', 'Area_Uganda', 'Area_Ukraine', 'Area_United Kingdom', 'Area_Uruguay', 'Area_Zambia', 'Area_Zimbabwe',
    'Item_Cassava', 'Item_Maize', 'Item_Plantains and others', 'Item_Potatoes', 'Item_Rice, paddy', 'Item_Sorghum', 'Item_Soybeans', 'Item_Sweet potatoes', 'Item_Wheat', 
    'Item_Yams', 'Year_1990', 'Year_1991', 'Year_1992', 'Year_1993', 'Year_1994', 'Year_1995', 'Year_1996', 'Year_1997', 'Year_1998', 'Year_1999', 'Year_2000', 'Year_2001',
    'Year_2002', 'Year_2004', 'Year_2005', 'Year_2006', 'Year_2007', 'Year_2008', 'Year_2009', 'Year_2010', 'Year_2011', 'Year_2012', 'Year_2013']
   
   
   df = pd.DataFrame(data["data"], columns=columns_for_df)
   json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
   data = np.array(json_list)
   prediction = service.predict(data)
   return prediction

@pipeline(enable_cache=False, settings={"docker":docker_settings})
def continuous_deployment_pipeline(
      data_path: str,
      min_accuracy: float = 0.92,
      workers: int = 1,
      timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT, 
):
      df = ingest_df(data_path=data_path)
      X_train, X_test, y_train, y_test = clean_df(df)
      model = train_model(X_train, X_test, y_train, y_test)
      r2_score, rmse = evaluate_model(model, X_test, y_test)
      deployment_decision = deployment_trigger(r2_score)
      mlflow_model_deployer_step(
            model=model,
            deploy_decision=deployment_decision,
            workers=workers,
            timeout=timeout,
      )

@pipeline(enable_cache=False, settings={"docker":docker_settings})
def inference_pipeline(pipeline_name: str, pipeline_step_name: str):
     data = dynamic_importer()
     service = prediction_service_loader(
          pipeline_name=pipeline_name,
          pipeline_step_name=pipeline_step_name,
          running=False,
     )
     prediction = predictor(service=service, data=data)
     #return prediction