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
    
    cp = st.number_input('Key in chest pain type (cp), 0:typical angina, 1:atypical angina, 2:non-anginal pain, 3:asyptomatic')
    
    thalachh = st.number_input('Key in maximum heart rate achieved (thalachh)')
    
    oldpeak = st.number_input('Key in Oldpeak')
    
    thall = st.number_input('Key in thall')
    
    submitted = st.form_submit_button("Submit")
    if submitted:
        X_new = [age,cp,thalachh,oldpeak,thall]
        outcome = model.predict(np.expand_dims(np.array(X_new),axis=0))
        outcome_dict = {0: 'Less chance of heart attack',
                        1: 'More chance of heart attack'}
        st.write(outcome_dict[outcome[0]])
