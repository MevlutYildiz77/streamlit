import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image


#sidebar
#st.sidebar.title("Car Price Prediction")
#st.sidebar.header("Sidebar header")
#hp_kW=st.sidebar.slider("hp_kW",5,350,2,1)
#age=st.sidebar.slider("age",0,15,1)
#km=st.sidebar.slider("km",0,350000,2,1)
#make_model_Audi_A1=st.sidebar.slider("A1",0,1,0)
#make_model_Audi_A2=st.sidebar.slider("A2",0,1,0)
#make_model_Audi_A3=st.sidebar.slider("A3",0,1,0)
#make_model_Opel_Astra=st.sidebar.slider("Astra",0,1,0)
#make_model_Opel_Corsa=st.sidebar.slider("Corsa",0,1,0)
#make_model_Opel_Insignia=st.sidebar.slider("Insignia",0,1,0)
#make_model_Renault_Clio=st.sidebar.slider("Clio",0,1,0)
#make_model_Renault_Duster=st.sidebar.slider("Duster",0,1,0)
#make_model_Renault_Espace=st.sidebar.slider("Escape",0,1,0)
#Gearing_Type_Automatic=st.sidebar.slider("Automatic",0,1,0)
#earing_Type_Manual=st.sidebar.slider("Manual",0,1,0)
#Gearing_Type_Semi_automatic=st.sidebar.slider("Semi_automatic",0,1,0)

#dataframe
st.write("# dataframes")
df = pd.read_csv("final_scout_not_dummy_09.csv", nrows=(100))
#df = pd.read_csv("final_scout_not_dummy_09.csv", nrows=(100))
#st.table(df.head())
#st.write(df.head()) #dynamic, you can sort, swiss knife
st.dataframe(df.head())#dynamic
#Project Example
import pickle
filename = 'last_model'
mesut_model = pickle.load(open(filename, 'rb'))
#st.table(df.head())

st.write(df.describe())

hp_kW = st.sidebar.number_input("hp_kW:",min_value=5, max_value=300)
age = st.sidebar.number_input("age:",min_value=0, max_value=20)
km = st.sidebar.number_input("km:",min_value=0, max_value=350000)
make_model_Audi_A1 = st.sidebar.number_input("make_model_Audi A1:",min_value=0, max_value=1)
make_model_Audi_A2 = st.sidebar.number_input("make_model_Audi A2:",min_value=0, max_value=1)
make_model_Audi_A3 = st.sidebar.number_input("make_model_Audi A3:",min_value=0, max_value=1)
make_model_Opel_Astra = st.sidebar.number_input("make_model_Opel Astra:",min_value=0, max_value=1)
make_model_Opel_Corsa = st.sidebar.number_input("make_model_Opel Corsa:",min_value=0, max_value=1)
make_model_Opel_Insignia = st.sidebar.number_input("make_model_Opel Insignia:",min_value=0, max_value=1)
make_model_Renault_Clio = st.sidebar.number_input("make_model_Renault Clio:",min_value=0, max_value=1)
make_model_Renault_Duster = st.sidebar.number_input("make_model_Renault Duster:",min_value=0, max_value=1)
make_model_Renault_Espace = st.sidebar.number_input("make_model_Renault Espace:",min_value=0, max_value=1)
Gearing_Type_Automatic = st.sidebar.number_input("Gearing_Type_Automatic:",min_value=0, max_value=1)
Gearing_Type_Manual = st.sidebar.number_input("Gearing_Type_Manual:",min_value=0, max_value=1)
Gearing_Type_Semi_automatic = st.sidebar.number_input("Gearing_Type_Semi-automatic:",min_value=0, max_value=1)
my_dict = {
    "hp_kW": hp_kW ,
    "age": age ,
    "km": km,
    'make_model Audi A1': make_model_Audi_A1,
 'make_model_Audi A2': make_model_Audi_A2,
 'make_model_Audi A3': make_model_Audi_A3,
 'make_model_Opel Astra': make_model_Opel_Astra,
 'make_model_Opel Corsa': make_model_Opel_Corsa,
 'make_model_Opel Insignia': make_model_Opel_Insignia,
 'make_model_Renault Clio' :make_model_Renault_Clio ,
 'make_model_Renault Duster':make_model_Renault_Duster ,
 'make_model_Renault Espace':make_model_Renault_Espace,
 'Gearing_Type_Automatic':Gearing_Type_Automatic,
 'Gearing_Type_Manual': Gearing_Type_Manual,
 'Gearing_Type_Semi-automatic': Gearing_Type_Semi_automatic
}
df1=pd.DataFrame.from_dict([my_dict])
st.table(df1)
if st.button("Predict"):
    pred = mesut_model.predict(df1)
    st.write(pred)
html_temp = """
<div style="background-color:green;padding:1.5px">
<h1 style="color:white;text-align:center;">Congradulations </h1>
</div><br>"""
st.markdown(html_temp,unsafe_allow_html=True)




