# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 15:27:22 2022

@author: End User
"""
import pickle
import os
import streamlit as st
import numpy as np


MODEL_PATH = os.path.join(os.getcwd(),'best_model.pkl')

with open(MODEL_PATH,'rb') as file:
    model = pickle.load(file)

with st.form("Patient's info: "):
    st.write("""
             # This app is to predict the possibility of a person to has heart attack
             """)
    st.write('Hit the information boxes to know more to know wether you have the possibility or not?')
    
    age = st.number_input('Key in your age')
    # sex = st.number_input('Key in sex, Female:0,Male:1')
    cp = st.number_input('Key in chest pain type (cp), 0:typical angina,1:atypical angina,2:non-anginal pain,3:asyptomatic')
    # trtbps = st.number_input('Key in resting blood pressure(trtbps)')
    # chol = st.number_input('Key in cholestrol readings')
    # fbs = st.number_input('Key in fasting blood sugar(fbs),0:False,1:True')
    # restecg = st.number_input('Key in resting electrocardiographic(restecg), 0:normal,1:ST-T wave abnormality,2:probable ventricular hypertrophy')
    thalachh = st.number_input('Key in maximum heart rate achieved(thalachh)')
    # exng = st.number_input('Key in exercise induced angina value(exng), 0:no,1:yes')
    oldpeak = st.number_input('Key in oldpeak')
    # slp = st.number_input('Key in slp')
    # caa = st.number_input('Key in caa')
    thall = st.number_input('Key in thall')
    
    submitted = st.form_submit_button("Submit")
    if submitted:
        X_new = [age,cp,thalachh,oldpeak,thall]
        outcome = model.predict(np.expand_dims(np.array(X_new),axis=0))
        outcome_dict = {0: 'Less chance of heart attack',
                        1: 'More chance of heart attack'}
        st.write(outcome_dict[outcome[0]])


































