# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 09:43:42 2022

@author: End User
"""
# About the dataset:
    # 1) Age: Age of the patient ---> continuous
    # 2) Sex: Sex of the patient ---> categorical
    # 3) cp: Chest pain type:    ---> categorical
             # Value 0 = typical angina
             # Value 1 = atypical angina
             # Value 2 = non-anginal pain
             # Value 3 = asymptomatic
    # 4) trtbps: resting blood pressure (in mm Hg) ---> continuous
    # 5) chol: cholesterol in mg/dl fetched via BMI sensor ---> continuous
    # 6) fbs: (fasting blood sugar > 120mg/dl)(1=true,0=false)--> categorical
    # 7) restecg: resting electrocardiographic results: --> categorical
             # Value 0 = normal
             # Value 1 = having ST-T wave abnormality
             # Value 2 = showing probable or definite left ventricular hypertrophy
    # 8) thalach: maximum heart rate achieved ---> continuous
    # 9) exng: exercise induced angina (1=yes,0=no)-->categorical
    # 10) oldpeak: --> continuous
    # 11) slp: [0,1,2] --> categorical
    # 12) caa: [0,1,2,3,4] --> categorical
    # 13) thall: [0,1,2,3] --> categorical
    # 14) output: [0,1] --> categorical

import os
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

from modules_for_heart_attack_prediction import functions,EDA

#%% Static
CSV_PATH = os.path.join(os.getcwd(),'heart.csv')
BEST_MODEL_PATH = os.path.join(os.getcwd(),'best_model.pkl')
BEST_PIPE_PATH = os.path.join(os.getcwd(),'best_pipe.pkl')

#%% EDA
# Step 1) Data Loading
df = pd.read_csv(CSV_PATH)
# Types of data:
    # continuous: age,trtbps,chol,thalach,oldpeak
    # categorical: sex,cp,fbs,restecg,exng,slp,caa,thall,output

# Step 2) Data inspection/visualization
df.info() # No NaNs
df.describe().T

    # To visualize the data:
con_columns = ['age','trtbps','chol','thalachh','oldpeak']
cat_columns = ['sex','cp','fbs','restecg','exng','slp','caa','thall','output']
eda = EDA()
eda.plot_graph(df,con_columns,cat_columns)

# Early hypothesis:
    # Inferences can be made when analysing the graph.

df.boxplot() # trtbps and chol have outliers

# To check the presence of NaNs and Duplicates
df.isna().sum() # There is no presence of NaNs
df.duplicated().sum() # There is 1 duplicated data in dataset.
df[df.duplicated()] # To show the duplicated data. The duplicated data at row 164.

# Step 3) Data cleaning
# List of things to be filtered:
    # 1) Duplicated data
    # 2) Remove rare reading of cholesterol level (>500mg/dl)
         # Readings at 500 mg/dl and more considered rare cases.

# Remove duplicated data
df = df.drop_duplicates()

# To ensure the presence of duplicates
df.duplicated().sum() # There is no duplicates in dataset.

# To remove rare reading of cholesterol level (>500 mg/dl)
df_copied = df.copy() # To copy original dataset from df into copied dataset,
                      # df_copied.
df_copied.columns = df.columns # To copy column names from df to df_copied

    # To ensure no cholesterol reading > 500 mg/dl
df_copied = df_copied[df_copied['chol']<500]
df_copied[df_copied['chol']>500] # To ensure the cholesterol readings is < 500
df_copied.describe().T # The maximum reading of cholesterol level is 417 mg/dl

# Step 4) Features selection
# Regression Analysis:
    # 1) Continuous vs categorical data --> Logistic Regression 
                                            # (to determine the accuracy,r/s in between)
    # 2) Categorical vs categorical data

# Continuous vs categorical data
for con in con_columns:
    print(con)
    lr = LogisticRegression()
    lr.fit(np.expand_dims(df_copied[con],axis=-1),df_copied['output'])
    print(lr.score(np.expand_dims(df_copied[con],axis=-1),df_copied['output']))

# Since thalachh (maximum heart rate achived),oldpeak and age achieved 70%,68%
# and 62% respectively trained by logistic regression against output,
# thus, those features will be selected for the subsequent steps.

# Categorical vs categorical data
func = functions()
for cat in cat_columns:
    print(cat)
    confusion_mat = pd.crosstab(df_copied[cat],df_copied['output']).to_numpy()
    print(func.cramers_corrected_stat(confusion_mat))

# Therefore, thall and cp are the two highest among all which achieved 
# correlation of 52% and 50% respectively. Hence, those two will be selected
# for the subsequent steps.

# Step 5) Data Pre-processing
# First, check whether MinMaxScaler or StandardScaler work best in combination 
# with which classifiers for the dataset.

X = df_copied.loc[:,['thalachh','oldpeak','age','thall','cp']]
y = df_copied['output']
X_train,X_test,y_train,y_test = train_test_split(X,y,
                                                 test_size=0.3,
                                                 random_state=123)

#%% Pipeline
# Logistic regression
step_mms_lr = Pipeline([('mms',MinMaxScaler()),
                      ('lr',LogisticRegression())])

step_ss_lr = Pipeline([('ss',StandardScaler()),
                      ('lr',LogisticRegression())])

# Random forest
step_mms_rf = Pipeline([('mms',MinMaxScaler()),
                      ('rf',RandomForestClassifier())])

step_ss_rf = Pipeline([('ss',StandardScaler()),
                      ('rf',RandomForestClassifier())])

# Decision tree
step_mms_dt = Pipeline([('mms',MinMaxScaler()),
                      ('dt',DecisionTreeClassifier())])

step_ss_dt = Pipeline([('ss',StandardScaler()),
                      ('dt',DecisionTreeClassifier())])

# KNeighbors
step_mms_knn = Pipeline([('mms',MinMaxScaler()),
                        ('knn',KNeighborsClassifier())])

step_ss_knn = Pipeline([('ss',StandardScaler()),
                      ('knn',KNeighborsClassifier())])

#%%
# Assign steps into pipelines
pipelines = [step_mms_lr,step_ss_lr,step_mms_rf,step_ss_rf,step_mms_dt,
             step_ss_dt,step_mms_knn,step_ss_knn]

for pipe in pipelines:
    pipe.fit(X_train,y_train)

best_accuracy = 0

# Model Evaluation
for i,model in enumerate(pipelines):
    print(model)
    print(model.score(X_test,y_test))
    if model.score(X_test,y_test) > best_accuracy:
        best_accuracy = model.score(X_test,y_test)
        best_pipeline = model

print('The best pipeline for this Heart Attack Dataset is {} with accuracy of {}'.format(best_pipeline,best_accuracy))

#%% Model fine tuning

# Step for Random Forest
step_rf = [('mmsscaler', MinMaxScaler()),
           ('rf',RandomForestClassifier())]

pipeline_rf = Pipeline(step_rf)

# Number of trees
grid_param = [{'rf':[RandomForestClassifier()],
               'rf__n_estimators':[10,100,1000],
               'rf__max_depth':[3,5,7,10,None],
               'rf__min_samples_leaf':np.arange(1,5)}]

gridsearch = GridSearchCV(pipeline_rf,grid_param,cv=5,verbose=1,n_jobs=-1)
best_model = gridsearch.fit(X_train,y_train)

#%%% Retrain model with selected parameters
step_mms_rf = Pipeline([('mms',MinMaxScaler()),
                      ('rf',RandomForestClassifier(n_estimators=10,
                                                    min_samples_leaf=4,
                                                    max_depth=7))])

step_mms_rf.fit(X_train,y_train)

with open(BEST_MODEL_PATH,'wb') as file:
    pickle.dump(step_mms_rf,file)

#%%
print(best_model.score(X_test,y_test))
print(best_model.best_index_)
print(best_model.best_params_)

# For Random Forest Classifier, the best number of estimator is 10, 
# max depth is 7 and min samples leaf is 4
# The accuracy dropped from 80% to 76%

# Packaging model
with open(BEST_PIPE_PATH,'wb') as file:
    pickle.dump(best_model,file)

#%% Model analysis
y_true = y_test
y_pred = best_model.predict(X_test)
print(classification_report(y_true,y_pred))
print(confusion_matrix(y_true,y_pred))
print(accuracy_score(y_true,y_pred))

#%% Discussion
# Dataset description:
    # 1) Patients' age have normal distribution with an average at 54 years old.
    # 2) The minimum and maximum resting blood pressure (trtbps) among the patients
         # at 94 mmHg and 200 mmHg respectively with average at 131 mmHg.
    # 3) The minimum and maximum cholesterol (chol) among the patients
         # at 126 mg/dl and 564 mg/dl respectively with average at 246 mg/dl.
    # 4) The patients maximum heart rate achieved distribution 
         # is a little bit negatively skewed. The minimum and maximum heart rate 
         # achieved (thalachh) among the patients at 71 BPM and 202 BPM 
         # respectively with average at 149 BPM.
    # 5) The oldpeak distribution is rightly skewed (positively skewed) with
         # minimum and maximum at 0 and 6.2 respectively. Meanwhile tha average
         # at 1.
    # 6) The maximum resting blood pressure at 200mmHg indicates extremely 
         # high blood pressure and leads to stroke
    # 7) For someone has cholesterol level of 564 mg/dl is possible especially
         # for someone with familial hypercholesterolemia.

# Discussion
# The best combination for this dataset is between MinMaxScaler and RandomForest
# The accuracy achieved is around 80%
# But after doing the the model fine tuning, the accuracy dropped to 76%
