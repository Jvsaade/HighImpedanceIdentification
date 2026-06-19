import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# 994 Classe 1; 104 Classe 0.
df = pd.read_csv("features.csv")
df.drop(columns=['Unnamed: 0'],inplace=True)

df0 = df[df['Classe'] == 0]
df1 = df[df['Classe'] == 1]

y = df['Classe']
X = df.drop(columns=['Classe'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

smote = SMOTE()

X_train, y_train = smote.fit_resample(X_train,y_train)