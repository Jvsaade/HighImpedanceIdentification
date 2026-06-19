import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from basicConfig import X_train, y_train, X_test, y_test

print(f"Formato X_train = {X_train.shape}")
print(f"Formato y_train = {y_train.shape}")

classe0 = y_train[y_train==0]
classe1 = y_train[y_train==1]

print(f"Tamanho classe 0: {len(classe0)}")
print(f"Tamanho classe 1: {len(classe1)}")


print()

pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC(kernel='rbf',class_weight='balanced'))])
result = pipe.fit(X_train, y_train).score(X_test, y_test)
print(f"Resultado da aplicação de Support Vector Machine (rbf): {result:.2%}")
y_pred = pipe.predict(X_test)
print(f"Classification report: {classification_report(y_test, y_pred)}")
disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred))
disp.plot()
plt.show()