from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from basicConfig import X, y

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

base_rf = RandomForestClassifier(
    n_estimators=300,
    min_samples_split=10,      
    min_samples_leaf=5,         
    max_features='sqrt',       
    class_weight='balanced'    
)

pipe = Pipeline([
    ('scaler', StandardScaler()), 
    ('randfor', base_rf)
])

pipe.fit(X_train, y_train)
result = pipe.score(X_test, y_test)
print(f"Score: {result:.2%}")

rf_model = pipe.named_steps['randfor']