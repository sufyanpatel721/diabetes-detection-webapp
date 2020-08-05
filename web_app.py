# this program detects if someone has diabetes or not using machine learning

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st
st.header('                 Diabetes Detection')
#create an title and subtitle
# st.write("""
# diabetes detection
# detect if someone has diabetes by using machine learning and python!
# """)

#open and display image
image = Image.open('bgimage.png')
st.image(image, caption='ML', use_column_width=True)

#Get the data
df = pd.read_csv('diabetes.csv')

#set the subheader on the webapp
# st.subheader('Data Information')
#
# #show the data as a table
# st.dataframe(df)
#
# #show statistics on the data
# st.write(df.describe())
#
# #show the data as a chart
# chart = st.bar_chart(df)

#split the data into independent x and dependent y variables
X = df.iloc[:, 0:8].values
y = df.iloc[:, -1].values

#split the data into training and and testing
X_train ,X_test, y_train ,y_test = train_test_split(X,y,test_size=0.25,random_state=0)

#get the features input from the users
def get_user_input():
    pregnancies = st.sidebar.slider('pregnancies',0,17,3)
    glucose = st.sidebar.slider('glucose',0,199,117)
    blood_pressure = st.sidebar.slider('blood_pressure',0,199,117)
    skin_thickness = st.sidebar.slider('skin_thickness',0,99,23)
    insulin = st.sidebar.slider('insulin',0.0,846.0,30.0)
    bmi = st.sidebar.slider('bmi',0.0,67.1,32.0)
    diabetes_pedigree_function = st.sidebar.slider('diabetes_pedigree_function',0.078,2.42,0.3725)
    age = st.sidebar.slider('age',21,81,29)

    user_data = ({
                'pregnancies' : [pregnancies],
                'glucose' : [glucose],
                'blood_pressure' : [blood_pressure],
                'skin_thickness' : [skin_thickness],
                'insulin' : [insulin],
                'bmi' : [bmi],
                'diabetes_pedigree_function' : [diabetes_pedigree_function],
                'age' : [age]
                })
#transform the data into a dataframe
    features = pd.DataFrame(user_data)
    return features


user_input = get_user_input()

st.subheader('User Input')
st.write(user_input)

RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(X_train,y_train)
#model accuracy score
st.subheader('Model test accuracy score!')
st.write(str(accuracy_score(y_test, RandomForestClassifier.predict(X_test)) * 100)+'%')


prediction = RandomForestClassifier.predict(user_input)

#set a subheader and display a classifier
st.subheader('Do you have diabetes: (1) means Yes (0) means No ')
st.write(int(prediction))
