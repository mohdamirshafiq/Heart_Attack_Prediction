# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 18:33:43 2022

@author: End User
"""
import scipy.stats as ss
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class functions():
    def __init__(self):
        pass
    
    def cramers_corrected_stat(self,confusion_matrix):
        """ calculate Cramers V statistic for categorial-categorial association.
            uses correction from Bergsma and Wicher, 
            Journal of the Korean Statistical Society 42 (2013): 323-328
        """
        chi2 = ss.chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum()
        phi2 = chi2/n
        r,k = confusion_matrix.shape
        phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
        rcorr = r - ((r-1)**2)/(n-1)
        kcorr = k - ((k-1)**2)/(n-1)
        return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))

class EDA():
    def __init__(self):
        pass
    
    def plot_graph(self,df,con_columns,cat_columns):
        
        for con in con_columns:
            plt.figure()
            sns.distplot(df[con])
            plt.show()
        
        for cat in cat_columns:
            plt.figure()
            sns.countplot(df[cat])
            plt.show()

























































