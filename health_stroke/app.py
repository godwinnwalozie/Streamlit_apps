import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from PIL import Image
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

sns.set_theme(font_scale=0.7, style="darkgrid")

master_df= pd.read_csv("C:/Users/godwi/GitHub/Streamlit_apps/health_stroke/master_df.csv")


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


def main():
    #-------------------------------Sisdebar-----------------------
    #with st.sidebar.container():
        #image = Image.open( )
        #st.image(image)
    st.sidebar.title(' Select Features')


    # getting data from user
    #st.sidebar.image = Image.open("") 

    gender = st.sidebar.selectbox('Whats your gender?',('Male', 'Female'))
    age = st.sidebar.number_input('Input Age (min is 25, max is 100)', key = 'int',max_value  =100,min_value = 25) #st.sidebar.slider('Age',0,50,100)         
    hypertension = st.sidebar.selectbox('Are you hpertensive? 0 = No : 1= Yes',(1, 0))          
    heart_disease = st.sidebar.selectbox('Any heart related disease ? 0 = No 1= Yes',(1, 0))                     
    work_type = st.sidebar.selectbox('Work type ?', ("Private" ,"Self-employed","Children","Govt_job ","Never_worked"))             
    Residence_type = st.sidebar.selectbox('Residence Type ?', ("Urban","Rural"))             
    avg_glucose_level= st.sidebar.number_input('Average Gloucose Level', min_value= 0 )          
    bmi = st.sidebar.number_input('Enter your current BMI', min_value= 0, max_value= 100)              
    smoking_status = st.sidebar.selectbox('Smoking status ?',("Never smoked" , "Unknown","formerly smoked","Smokes","Never_smoked")) 
    st.sidebar.markdown("***")
    age = round(age,2)



    #st.title('Stroke Disease Prediction with Machine Learning')
    st.markdown("<h1 style='text-align: left; color: grey;'>Stroke Disease Prediction with Machine Learning</h1>", unsafe_allow_html=True)
    st.write(" **By Godwin Nwalozie : July 2022** ")
    image = Image.open("C:/Users/godwi/Pictures/strokeapp_headerimage.png") 
    st.image(image)


    with st.container():
        st.info(" ###### This Machine Learning algorithm tries to predict whether a patient is likely to get stroke based on the feature\
                            parameters like gender, age, various diseases, and smoking status. Each row in the \
                            data provides relavant information about the patient \n SOURCE DATA :\
                            ***https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset*** ")


    if hypertension == 1:
        hypertension = "Yes"
    else:
        "No"
        
    if heart_disease == 1:
        heart_disease = "Yes"
    else: "No"
    #st.markdown("***")
    
    #st.write ('### Stroke Disease Prediction App')  
    with st.container():
    # Store inputs into dataframe
        features = {"gender": gender,"age":age,"hypertension": hypertension, "heart_disease":heart_disease, "work_type" : work_type,"Residence_type" : Residence_type,
                "avg_glucose_level" : avg_glucose_level, "bmi" : bmi,"smoking_status": smoking_status }
        show = pd.DataFrame(features, index= [0])
        upper  = [i.upper() for i in show]
        show.columns = upper

        #st.markdown("***")
        

        row_count = len(master_df)
        col_count_ini = len(master_df.columns)
        col_count = len(master_df.columns)-3
        st.markdown("<h4 style='text-align: left; color: grey;'> Info about Dataset and Model </h1>", unsafe_allow_html=True)
        
        
        col1, col2, col3,col4,col5 = st.columns(5)
        col1.metric("Estimator","RFClassifier", "+")
        col2.metric("Model Prediction Accuracy", "93%", "<>")
        col3.metric("Number of rows trained", row_count, "<>")
        col4.metric("Initial No of columns", col_count_ini, "<>")
        col5.metric("No of columns trained", col_count, "<>")
    

        
        st.info (' ###### **Your Selected Features**')
        st.table(show)



    inputed = pd.DataFrame(data = features, index = [0])

    stroke_model = pickle.load(open('C:/Users/godwi/GitHub/Streamlit_apps/health_stroke/estimator_pkl', 'rb'))      
    
    le = LabelEncoder()
    inputed.gender =le.fit_transform(inputed.gender)
    inputed.work_type =le.fit_transform(inputed.work_type)
    inputed.Residence_type =le.fit_transform(inputed.Residence_type)
    inputed.smoking_status =le.fit_transform(inputed.smoking_status)
    inputed.hypertension =le.fit_transform(inputed.hypertension)
    inputed.heart_disease =le.fit_transform(inputed.heart_disease)
    inputed.bmi = inputed.bmi.apply(lambda x: round(x,2))



    prediction = stroke_model.predict(inputed)
    if prediction [0] == 0:
        prediction =  "Low Risk " 
        #st.balloons()
    else:
        prediction = "High Risk"
    if st.button("Click to Make Prediction"):
        # Output prediction
        st.success(f"Prediction : {prediction}")
    else:
        st.write('###### **Prediction display here !!**')
    st.markdown("***")

    

    

if __name__ == "__main__":
    
    
    
      main()
      
with st.container():
        st.markdown("<h5 style='text-align:left; color: black;'>Preprocessed Data Correlation & Confusion Matrix </h4>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            
            #bmi age correlation
            fig,ax = plt.subplots(figsize = (6,4))
            sns.regplot(data = master_df, x= "bmi", y= "age",marker="x")
            ax.set(title ="Correlation between Age and BMI")
            st.write(fig)
            
            
            st.markdown("***")
            
            
            # chart for confusion metrix
            conf_max_df= pd.read_csv ("C:/Users/godwi/GitHub/Streamlit_apps/health_stroke/conf_max_df.csv")
            fig,ax = plt.subplots(figsize = (6,2.6))
            sns.heatmap(conf_max_df/np.sum(conf_max_df) ,xticklabels = True, annot =True,fmt =".2%",
                        ax = ax,linewidths=0.2, linecolor='grey',)
            ax.set(title ="Confusion Matrix")
            st.write(fig)
            
            st.markdown("***")
            
                
        with col2:
            
            
            # chart for feature correlation
        
            fig,ax =plt.subplots(figsize = (6,3.4))
            feature_check =sns.heatmap(master_df.corr(), cmap = "Greens", annot = True,linewidths=0.2, linecolor='grey');
            ax.set(title ="Feature Correlation")
            st.write(fig)
            st.markdown("***")
            
            # chart for heart diseaase
            disease_check = pd.crosstab(master_df.gender, master_df.heart_disease).rename({0: "No", 1:"Yes"}, axis = 1)
            fig, ax =plt.subplots(figsize = (6,3))
            #disease_check.plot( kind = 'bar', color = ('teal',"blueviolet"), ax=ax)
            sns.heatmap(data = disease_check, annot = True, fmt ="2d", cmap = "Blues",linewidths=0.2, linecolor='grey' )
            ax.set( xlabel = 'Has Heart disease ?')
            ax.set(title ="Heart Disease by Gender")
            st.write(fig)


        st.write(' Thank you for visiting :-) '  )
        image = Image.open("C:/Users/godwi/Pictures/mazi2.png")
        st.image(image, width= 130)
        st.write("**Godwin**")

        st.write('Visit my page on Kaggle : https://www.kaggle.com/godwinnwalozie/code')
        st.write('Visit my Github page :https://github.com/godwinnwalozie')