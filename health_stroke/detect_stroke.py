from os import X_OK
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump, load
import pickle
import  joblib
from PIL import Image
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer


st.set_page_config(layout="wide")
m = st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #0099ff;
    color:#ffffff;
}
div.stButton > button:hover {
    background-color: #00ff00;
    color:#ff0000;
    }
</style>""", unsafe_allow_html=True)




st.sidebar.header(' Select Personal Features')

# getting data from user
gender = st.sidebar.selectbox('Whats your gender?',('Male', 'Female'))
age = st.sidebar.slider('Age',0,50,100)         
hypertension = st.sidebar.selectbox('Are you hpertensive? 0 = No 1= Yes',(1, 0))          
heart_disease = st.sidebar.selectbox('Any heart related disease ? 0 = No 1= Yes',(1, 0))           
ever_married = st.sidebar.selectbox('Ever married ?', ("Yes","No"))             
work_type = st.sidebar.selectbox('Work type ?', ("Private" ,"Self-employed","Children","Govt_job ","Never_worked"))             
Residence_type = st.sidebar.selectbox('Residence Type ?', ("Urban","Rural"))             
avg_glucose_level= st.sidebar.number_input('Average Gloucose Level' )          
bmi = st.sidebar.number_input('Your bmi')              
smoking_status = st.sidebar.selectbox('Smoking status ?',("Never smoked" , "Unknown","formerly smoked","Smokes","Never_smoked")) 
st.sidebar.markdown("***")
age = round(age,2)
master_df= pd.read_csv("C:/Users/godwi/GitHub/Streamlit_apps/health_stroke/df.csv")


st.markdown("<h2 style='text-align:left; color: grey;'>Stroke Disease Prediction with Machine Learning </h2>", unsafe_allow_html=True)
      
image = Image.open('C:/Users/godwi/GitHub/Streamlit_apps/health_stroke/header_image.png') 
st.image(image)
#st.write ('### Stroke Disease Prediction App')  

# Store inputs into dataframe
features = {"gender": gender,"age":age,"hypertension": hypertension, "heart_disease":heart_disease, "ever_married" :ever_married,
            "work_type" : work_type,"Residence_type" : Residence_type,
            "avg_glucose_level" : avg_glucose_level, "bmi" : bmi,"smoking_status": smoking_status }

show = pd.DataFrame(features, index= [0])
upper  = [i.upper() for i in show]
show.columns = upper


st.write (' #### The features you selected')
st.table(show)

inputed = pd.DataFrame(data = features, index = [0])

stroke_model = pickle.load(open('C:/Users/godwi/GitHub/Streamlit_apps/health_stroke/estimator_pkl', 'rb'))      
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
inputed.gender =le.fit_transform(inputed.gender)
inputed.ever_married =le.fit_transform(inputed.ever_married)
inputed.work_type =le.fit_transform(inputed.work_type)
inputed.Residence_type =le.fit_transform(inputed.Residence_type)
inputed.smoking_status =le.fit_transform(inputed.smoking_status)
inputed.bmi = inputed.bmi.apply(lambda x: round(x,2))

prediction = stroke_model.predict(inputed)

if prediction [0] == 0:
    prediction =  "Low Risk " 
    #st.balloons()
else:
    prediction = "High Risk"

if st.button("Submit"):
   
    # Output prediction
    st.success(f"Prediction : {prediction}")
else:
    st.write('#### Prediction will be display here !!')
    
