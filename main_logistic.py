import pandas as pd
import joblib
data = pd.read_csv(r"C:\Users\chint\vs code python\churn-prediction-ml\telco_dataset.csv")
data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})
data.drop('customerID',axis=1,inplace=True)
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data['TotalCharges'].fillna(data['TotalCharges'].mean(), inplace=True)
data = pd.get_dummies(data, drop_first=True)
data.fillna(0, inplace=True)
data = data[['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']]
x=data.drop('Churn', axis=1)    
y=data['Churn']
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print("accuracy:", model.score(x_test, y_test))
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")