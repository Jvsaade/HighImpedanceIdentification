import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from basicConfig import X, y

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y
)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95)),
    ('clf', LogisticRegressionCV(
        cv=5,
        solver='lbfgs',
        max_iter=1000,
        l1_ratios=(0,),                    
        scoring='accuracy',
        use_legacy_attributes=False         
    ))
])

pipeline.fit(X_train, y_train)

best_C = pipeline.named_steps['clf'].C_

print(f"Best C found: {best_C:.3f}")     
    
print(f"Auto-tuned C, Test Score: {pipeline.score(X_test, y_test):.3%}")

print(f"\nTraining set class distribution: {np.bincount(y_train)}")
print(f"Test set class distribution: {np.bincount(y_test)}")

from sklearn.metrics import classification_report
y_pred = pipeline.predict(X_test)
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1']))