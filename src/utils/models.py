import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier

def train_model(X_train,y_train,n_est,min_sample_split,n_jobs,ram_st):
    model = RandomForestClassifier(
        n_estimators= n_est,min_samples_split = min_sample_split,n_jobs = n_jobs,random_state = ram_st
    )
    model.fit(X_train,y_train)
    return model