import json
import numpy as np
import pandas as pd
import streamlit as st
from pipelines.deployment_pipeline import prediction_service_loader
from run_deployment import run_deployment


def main():
    st.title("End to End Crop Yield Prediction Pipeline with ZenML")

    # Define your list of columns

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
    rainfall = st.number_input("Enter Average Rainfall (mm/year):")
    pesticides = st.number_input("Enter Pesticides (tonnes):")
    temperature = st.number_input("Enter Average Temperature:")
    selected_year = st.selectbox("Select a Year:", columns_for_df[114:137])  # Select from Year columns
    
    # Create selectboxes for 'Area' and 'Item'
    selected_area = st.selectbox("Select an Area:", columns_for_df[3:104])  # Select from Area columns
    selected_item = st.selectbox("Select an Item:", columns_for_df[104:114])  # Select from Item columns
    
    # Create a dictionary to store the selected inputs
    selected_data = {col: 0 for col in columns_for_df}
    selected_data['average_rain_fall_mm_per_year'] = rainfall
    selected_data['pesticides_tonnes'] = pesticides
    selected_data['avg_temp'] = temperature
    selected_data[selected_year] = 1
    selected_data[selected_area] = 1
    selected_data[selected_item] = 1
    
    # Create a DataFrame with the selected inputs
    data_df = pd.DataFrame(selected_data, index=[0])
    
    # Display the selected inputs
    st.write("Selected Inputs:")
    st.write(data_df)

    if st.button("Predict"):
        service = prediction_service_loader(
        pipeline_name="continuous_deployment_pipeline",
        pipeline_step_name="mlflow_model_deployer_step",
        running=False,
        )
        if service is None:
            st.write(
                "No service could be found. The pipeline will be run first to create a service."
            )
            run_deployment()
        json_list = json.loads(json.dumps(list(data_df.T.to_dict().values())))
        data = np.array(json_list)
        pred = service.predict(data)
        st.write(pred)
       
if __name__ == "__main__":     
   main()    

    