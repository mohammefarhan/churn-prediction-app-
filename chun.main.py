import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
from xgboost import XGBClassifier
from joblib import dump

# Load data
churn_data = pd.read_csv('Churn_Modelling.csv')

X = churn_data.drop(['RowNumber', 'CustomerId', 'Surname', 'Exited'], axis=1)
y = churn_data['Exited']

# Label encoding
le_geo = LabelEncoder()
le_gender = LabelEncoder()
X['Geography'] = le_geo.fit_transform(X['Geography'])
X['Gender'] = le_gender.fit_transform(X['Gender'])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=99)

# Model
model = XGBClassifier()
model.fit(X_train, y_train)

# Predictions
preds = model.predict(X_test)

# Accuracy
print("Accuracy:", metrics.accuracy_score(y_test, preds))
dump(model, 'xgboost_churn.joblib')