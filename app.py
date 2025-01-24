import pandas as pd
import tensorflow as tf
import streamlit as st
import pickle
model = tf.keras.models.load_model('model.h5')

with open('lable_encoder.pkl','rb') as file:
    le = pickle.load(file)

with open('ohe.pkl','rb') as file:
    ohe = pickle.load(file)

with open('StandardScaler.pkl','rb') as file:
    sc = pickle.load(file)


st.title('Churn Prediction')

geography = st.selectbox('Geography',ohe.categories_[0])
gender = st.selectbox('Gender',le.classes_)
age = st.slider('Age',18,92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure',0,10)
num_of_products = st.slider('Number of Products',1,4)
has_cr_card = st.selectbox('Has credit Card', [0,1])
is_active_member = st.selectbox("Is Active Member",[0,1])


input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [le.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

geo_encoded = ohe.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded,columns = ohe.get_feature_names_out(['Geography']))

input_data = pd.concat([input_data.reset_index(drop = True),geo_encoded_df],axis=1)

input_scaled = sc.transform(input_data)

#predict
prediction = model.predict(input_scaled)
prediction_proba = prediction[0][0]

st.write(prediction_proba)

if prediction_proba > 0.5:
    st.write("gone")
else:
    st.write("safe")


