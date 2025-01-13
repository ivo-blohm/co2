import streamlit as st 
import pandas as pd
import pickle
from sklearn.tree import DecisionTreeRegressor

def co2_mean():
    data = pd.read_csv("co2_data_merged_eng.csv")
    mean=data["CO2"].mean()
    return mean

def load_data():
    data = pd.read_csv("co2_data_merged_eng.csv")
    del(data["CO2"])
    return data.sort_values("Zipcode",ascending=True)

def load_model():
    file_path = "tree_model.sav"
    # Modell mit pickle laden
    with open(file_path, "rb") as f:
        loaded_model = pickle.load(f)
    return loaded_model


data = load_data()
model = load_model()
co2_mean = co2_mean()

st.header("Predict CO2 Emissions")

col1_1, col1_2, col1_3, col1_4 = st.columns([1,1,1,1])
col2_1, col2_2, col2_3, col2_4 = st.columns([1,1,1,1])

nutzung = col1_1.selectbox("Utilization", data["Utilization"].unique())    
energieträger = col1_2.selectbox("Energy Source", data["Energy_Source"].unique())   
kanton = col1_3.selectbox("Canton", data["Canton"].unique())   
plz = col1_4.selectbox("Zipcode", data["Zipcode"].unique())    

ebf = col2_1.slider("Heated Area (m²)", data["Heated_Area"].min(), data["Heated_Area"].max())
jahr = col2_2.slider("Year of Measurement", data["Year_of_Measurement"].min(), data["Year_of_Measurement"].max())
baujahr = col2_3.slider("Year of Construction", data["Year_of_Construction"].min(), data["Year_of_Construction"].max())
erneuerbar = col2_4.slider("Renewable Energy (%)", data["Percentage_Renewable_Energy"].min(), data["Percentage_Renewable_Energy"].max())

last=pd.DataFrame.from_dict({"Utilization":[nutzung],
                             "Energy_Source": [energieträger],
                             "Canton":[kanton],
                             "Zipcode":[plz],
                             "Heated_Area":[ebf],
                             "Year_of_Measurement":[jahr],
                             "Year_of_Construction":[baujahr],
                             "Percentage_Renewable_Energy":[erneuerbar]
                             })
data = pd.concat([data, last], axis=0)
data = pd.get_dummies(data)
preds = model.predict(data)
pred = round(preds[-1],2)

delta = round((pred-co2_mean)/co2_mean,2)

formatted_delta = "{:.0%}".format(delta)

st.metric(label="CO2 Prediction", value=pred, delta=formatted_delta)
