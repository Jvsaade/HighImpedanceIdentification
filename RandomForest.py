from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from basicConfig import X_train, X_test, y_train, y_test

import time

start_time = time.perf_counter()

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

y_pred = pipe.predict(X_test)

end_time = time.perf_counter()

print(f"Score: {result:.2%}")

rf_model = pipe.named_steps['randfor']

disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred))
disp.plot()
plt.show()

# Calculate and print execution time
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.6f} seconds")