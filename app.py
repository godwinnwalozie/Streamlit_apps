import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from PIL import Image
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
import os




# css styling

st.set_page_config(layout="wide")

# submit button style
m = st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #0068C9 ;
    color:#ffffff;
}
div.stButton > button:hover {
    background-color: #00802b;
    color:#ccffff;
    }
</style>""", unsafe_allow_html=True)




# Remove whitespace from the top of the page and sidebar
st.markdown("""
        <style>
               .css-18e3th9 {
                    padding-top: 2.5rem;
                    padding-bottom: 5rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
               .css-hxt7ib{
                    padding-top: 3rem;
                    padding-right: 2rem;
                    padding-bottom: 2rem;
                    padding-left: 2rem;
                    color: rgb(247 247 250);
                }
                .css-1siy2j7 {
                    background-color: rgb(0 104 201);
                }
                .css-zx8yrj {
                    background-color: #f2f7fe;
                }
                .css-qrbaxs {
                    font-size: 14px;
                    color: rgb(241 241 241);
                }
                .st-br {
                    color: black;
                }
                .css-gdzsw5 {
                    font-weight: 600;
                }
                .css-a51556 {
                    font-weight: 900;
                    color: ash;
            }         
        </style>
        """, unsafe_allow_html=True)


# Hide Image Full Screen
hide_img_fs = '''
<style>
button[title="View fullscreen"]{
    visibility: hidden;}
</style>
'''
st.markdown(hide_img_fs, unsafe_allow_html=True)



# seting path
path = os.path.dirname("/Users/godwi/GitHub/streamlit_stroke/data/")
with st.container():
    
    #def file_selector(folder_path='/Users/godwi/GitHub/streamlit_stroke'):
        #filenames = os.listdir(folder_path)
        #selected_filename = st.selectbox('Select a file', filenames)
        #return os.path.join(folder_path, selected_filename)

    #filename = file_selector()
    #st.write('You selected `%s`' % filename)
    
    #st.write(path)
      
    header_image = (path +"/header_image.png")
    #st.write(header_image)
    image_header = Image.open(header_image)
    st.image(image_header)
    

st.write(" #### **Developed by : Godwin Nwalozie**")
#st.markdown("<h1 style='text-align: left; color: black;'>Stroke Disease Prediction</h1>", unsafe_allow_html=True)
#st.write("#### Godwin Nwalozie July 2022 ")
#st.markdown("***")
#st.markdown("***")
with st.container():
    st.info(" ###### Stroke is a potentially fatal medical condition that needs to be addressed.\nBased on criteria such as gender, age, heart disease, and smoking status, \
        this Machine Learning system attempts to predict whether a patient is likely to have a stroke. \
            Each row of data contains pertinent information about the patient.  \n **Data Source** :\
                            ***https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset*** ")

st.markdown("") 

    #st.write ('### Stroke Disease Prediction App')

    # The dataset for the estimator
dataframe = (path+ "/master_df.csv")
master_df= pd.read_csv(dataframe)

    
    # Store inputs into dataframe
select_features = (path+ "/Select_features.png") 
select_features = Image.open(select_features)
st.sidebar.image(select_features)
# st.sidebar.write('### Select Features')
gender = st.sidebar.selectbox('Whats your gender?',('Male', 'Female'))
age = st.sidebar.number_input('Input Age ', key = 'int',max_value  =100,min_value = 18) #st.sidebar.slider('Age',0,50,100)
hypertension = st.sidebar.selectbox('Are you hpertensive? ',("Yes", "No"))
heart_disease = st.sidebar.selectbox('Any heart related disease ? ',("Yes", "No"))
#work_type = st.sidebar.selectbox('Work type ?', ("Private" ,"Self-employed","Children","Govt_job ","Never_worked"))
Residence_type = st.sidebar.selectbox('Residencial Type ', ("Urban","Rural"))
avg_glucose_level= st.sidebar.number_input('Average Gloucose Level', min_value= 0 , max_value=140)
bmi = st.sidebar.number_input('Enter your current BMI', min_value= 0, max_value= 100)
smoking_status = st.sidebar.selectbox('Smoking status',("Never smoked" , "Unknown","formerly smoked","Smokes","Never_smoked"))
#st.sidebar.markdown("***")




features = {"gender": gender,"age":age,"hypertension": hypertension, "heart_disease":heart_disease, "Residence_type" : Residence_type,
                "avg_glucose_level" : avg_glucose_level, "bmi" : bmi,"smoking_status": smoking_status }
show = pd.DataFrame(features, index= [0])
show.age= show.age.astype('int')
upper  = [i.capitalize() for i in show]
show.columns = upper

row_count = len(master_df)
col_count_ini = len(master_df.columns)
col_count = len(master_df.columns)-3
st.markdown("<h4 style='text-align: left; color: chocolate;'> Details of the trained dataset </h4>", unsafe_allow_html=True)

with st.container():
    col1, col2, col3,col4,col5 = st.columns(5)
    col1.metric("Number of rows trained", row_count, "<>")
    col2.metric("Initial No of columns", col_count_ini, "<>")
    col3.metric("No of columns trained", col_count, "-")
    col4.metric("Estimator","RFClassifier", "+")
    col5.metric("Model Prediction Accuracy", "93%", "<>")
    
    
#st.markdown("***")

with st.container():
    
    st.markdown("<h4 style='text-align: left; color: chocolate;'> Your selected features </h4>", unsafe_allow_html=True)
    st.table(show)
    

inputed = pd.DataFrame(data = features, index = [0])
le = LabelEncoder()
inputed.gender =le.fit_transform(inputed.gender)
#inputed.work_type =le.fit_transform(inputed.work_type)
inputed.Residence_type =le.fit_transform(inputed.Residence_type)
inputed.smoking_status =le.fit_transform(inputed.smoking_status)
inputed.hypertension =le.fit_transform(inputed.hypertension)
inputed.heart_disease =le.fit_transform(inputed.heart_disease)
#inputed.bmi = inputed.bmi.apply(lambda x: round(x,2))
     

trained_model = (path+ "/estimator_pkl")
stroke_model = pickle.load(open(trained_model, 'rb'))


prediction = stroke_model.predict(inputed)
if prediction [0] == 0:
    prediction =  "Class [0] LOW RISK" 
        #st.balloons()
else:
    prediction = "Class [1] HIGH RISK" 
    
if st.button("click to make prediction"):
           
    # Output prediction
    st.write(f" ##### The Prediction is a {prediction} ")
    #st.markdown("***")    
else:
    pred_smiley =(path+ "/smiley_click.png") 
    pred_smiley = Image.open(pred_smiley)
    st.image(pred_smiley,width= 400)
    #st.write('###### **Prediction display here !!**')
    
    
sns.set_theme(font_scale=0.84, style="darkgrid")
st.markdown("***")
#if __name__ == "__main__":
      #main()

with st.container():
    #master_df= pd.read_csv("C:/Users/godwi/GitHub/streamlit_app_stroke_precdict/data/master_df.csv")
    st.markdown("<h4 style='text-align:left; color: chocolate;'>Exploratory Data Analysis & Confusion Matrix Report</h4>",
                    unsafe_allow_html=True)


    
    
    col1, col2 = st.columns(2)
    with col1:
        
        fig, ax =plt.subplots(figsize = (6,4.1))
        gender_stat = master_df.gender.value_counts().to_frame()
        #disease_check.plot( kind = 'bar', color = ('teal',"blueviolet"), ax=ax)
        sns.barplot(data = gender_stat, x = gender_stat.index, y = gender_stat['gender'] ,\
        capsize = 0.05 ,ax=ax)
        ax.set(title ="Gender Distribution", xlabel ='gender', ylabel ='count')
        st.write(fig)
        
                
        st.markdown("***")
        # chart for heart diseaase
        disease_check = pd.crosstab(master_df.gender, master_df.heart_disease).rename({0: "No", 1:"Yes"}, axis = 1)
        fig, ax =plt.subplots(figsize = (7,7))
        #disease_check.plot( kind = 'bar', color = ('teal',"blueviolet"), ax=ax)
        sns.heatmap(data = disease_check, annot = True, fmt ="2d", cmap = "Blues",linewidths=0.4, linecolor='grey' )
        ax.set(title ="Heart Disease by Gender")
        st.write(fig)
        #st.markdown("***")
        
        st.markdown("***")
        #bmi age correlation
        fig,ax = plt.subplots(figsize = (6,4))
        sns.regplot(data = master_df, x= "bmi", y= "age",marker="x")
        ax.set(title ="Correlation between Age and BMI")
        st.write(fig)
        

            
            
        #st.markdown("***")
        
        with col2:
            # Age category Plot
            age_hyper = master_df.loc[:,["age","stroke"]]
            age_hyper.stroke = age_hyper.stroke.apply(lambda x: "Yes" if x == 1 else "No" )
            age_hyper['age_cat'] = age_hyper.age.apply(lambda x :  "0-2" if 0 <= x<2 else
                                                "2-5" if 2<= x<= 5 else
                                                "6-13" if 5< x< 13 else
                                                "13-18" if 13<= x< 18 else
                                                "18-30" if 18<= x< 30 else
                                                "30-40" if 30<= x< 40 else
                                                "40-50" if 40<= x< 50 else
                                                "50-65" if 50<= x< 65 else
                                                "65+" if x>= 65 else "not known")

            pivot_age = age_hyper.pivot_table(index = 'age_cat', columns='stroke', values="age", aggfunc= 'count')
            fig,ax = plt.subplots(figsize = (5,3))
            pivot_age.plot(kind = 'bar', ax = ax, fontsize = 8)
            ax.set_title("Stroke by Age Category", fontsize = 8)
            plt.legend(fontsize = 7, loc = "upper left")
            st.write(fig)
            #st.markdown("***")
            
            
            st.markdown("***")
            # Feature correlation
            fig,ax =plt.subplots(figsize = (15,14.1))
            feature_check =sns.heatmap(master_df.corr(), cmap = 'Greens', annot = True, linewidths=0.3,\
            linecolor='grey', ax = ax, annot_kws={'size': 15})
            ax.set_title ("Feature Correlation", fontsize = 20)
            ax.tick_params(labelsize=16)
            st.write(fig)
            
            
            st.markdown("***")
             # chart for confusion metrix            
            confusion = (path+ "/conf_max_df.csv") 
            conf_max_df= pd.read_csv (confusion).rename({0 : "No", 1 : "Yes"}, axis = 0).\
            rename({"Low Risk":"No", "High risk" : "Yes"}, axis =1 )
            fig,ax = plt.subplots(figsize = (12,7.5))
            sns.heatmap(conf_max_df ,xticklabels = True, annot =True, ax = ax,linewidths=0.2, \
            linecolor='grey',fmt = "2d",cmap = "Blues",annot_kws={'size': 15})
            ax.set_title ("Confusion Matrix (Truth Table)", fontsize = 17)
            ax.set_xlabel("Predicted Label",fontsize = 14)
            ax.set_ylabel("Actual Label",fontsize = 14)
            ax.tick_params(labelsize=15)
            st.write(fig)
            




            


st.markdown("***")
with st.container():

    st.text("""𝑾𝒊𝒕𝒉𝒐𝒖𝒕 𝒅𝒂𝒕𝒂 𝒚𝒐𝒖’𝒓𝒆 𝒋𝒖𝒔𝒕 𝒂𝒏𝒐𝒕𝒉𝒆𝒓 𝒑𝒆𝒓𝒔𝒐𝒏 𝒘𝒊𝒕𝒉 𝒂𝒏 𝒐𝒑𝒊𝒏𝒊𝒐𝒏.” 𝑬𝒅𝒘𝒂𝒓𝒅𝒔 𝑫𝒆𝒎𝒊𝒏𝒈, 𝑺𝒕𝒂𝒕𝒊𝒔𝒕𝒊𝒄𝒊𝒂𝒏 """)    
 
    git='Visit my Git [link](https://github.com/godwinnwalozie)'
    st.markdown(git,unsafe_allow_html=True)
    
    kaggle='Find me on Kaggle [link](https://www.kaggle.com/godwinnwalozie/code)'
    st.markdown(kaggle,unsafe_allow_html=True)
