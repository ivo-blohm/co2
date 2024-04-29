import streamlit as st 
import pandas as pd
import pickle
from sklearn.tree import DecisionTreeRegressor

def co2_mean():
    data = pd.read_csv("co2_data_merged.csv")
    mean=data["CO2"].mean()
    return mean

def load_data():
    data = pd.read_csv("co2_data_merged.csv")
    del(data["CO2"])
    return data.sort_values("Postleitzahl",ascending=True)

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

nutzung = col1_1.selectbox("Nutzung", data["Nutzung"].unique())    
energieträger = col1_2.selectbox("Energieträger", data["Energieträger"].unique())   
kanton = col1_3.selectbox("Kanton", data["Kanton"].unique())   
plz = col1_4.selectbox("Postleitzahl", data["Postleitzahl"].unique())    

ebf = col2_1.slider("EBF", data["EBF"].min(), data["EBF"].max())
jahr = col2_2.slider("Jahr", data["Jahr"].min(), data["Jahr"].max())
baujahr = col2_3.slider("Baujahr", data["Baujahr"].min(), data["Baujahr"].max())
erneuerbar = col2_4.slider("% Erneuerbarer Energie", data["Anteil_Erneuerbare_Energie"].min(), data["Anteil_Erneuerbare_Energie"].max())

last=pd.DataFrame.from_dict({"Nutzung":[nutzung],
                             "Energieträger": [energieträger],
                             "Kanton":[kanton],
                             "Postleitzahl":[plz],
                             "EBF":[ebf],
                             "Jahr":[jahr],
                             "Baujahr":[baujahr],
                             "Anteil_Erneuerbare_Energie":[erneuerbar]
                             })
data = pd.concat([data, last], axis=0)
data = pd.get_dummies(data)
preds = model.predict(data)
pred = round(preds[-1],2)

delta = round((pred-co2_mean)/co2_mean,2)

formatted_delta = "{:.0%}".format(delta)

st.metric(label="CO2 Prediction", value=pred, delta=formatted_delta)
