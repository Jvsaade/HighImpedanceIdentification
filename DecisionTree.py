from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from basicConfig import X_train, y_train, X_test, y_test

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('decisionTree', DecisionTreeClassifier())
])

pipe.fit(X_train, y_train)
result = pipe.score(X_test, y_test)
y_pred = pipe.predict(X_test)

print(f"Score: {result:.2%}")

dt_model = pipe.named_steps['decisionTree']

plt.figure(figsize=(12, 8))

plot_tree(
    dt_model,
    feature_names=X_train.columns,
    class_names=dt_model.classes_.astype(str)
)

plt.show()