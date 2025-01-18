# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC 

#  Load the Dataset
data_set = pd.read_csv(r"C:\SMEC\Data_Science\Project\DATASET\heart_disease_dataset.csv")
print(data_set.columns)
data_set.dropna(inplace=True)
print(data_set)

sns.boxplot(data=data_set)
plt.show()

correlation=data_set.corr(numeric_only=True)
sns.heatmap(correlation,annot=True)
plt.show()
df1=data_set.select_dtypes(exclude=['object'])
for coloumn in df1:
    plt.figure(figsize=(17,1))
    sns.boxplot(data=df1,x=coloumn)
print(df1)
plt.show()

# droping thge unwanted coloum
x=data_set.drop(columns=["Heart Disease","Gender","Smoking","Alcohol Intake","Family History","Diabetes","Obesity","Exercise Induced Angina"],axis=1)
y=data_set['Heart Disease']
print(x)
print(y)

# changing char to int with onehotencode
column_transformer = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [-1])], remainder='passthrough')
x=np.array(column_transformer.fit_transform(x))
print(x)

sns.scatterplot(x="Cholesterol",y="Heart Disease",data=data_set)
plt.show()

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
model1=LogisticRegression()
model1.fit(X_train,y_train)
y_pred=model1.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)*100
print("Model accuracy:",accuracy)

# "Support vector classifier" 
classifier = SVC(kernel='linear', random_state=0)
classifier.fit(X_train, y_train)

# Predicting the test set result
y_pred1= classifier.predict(X_test)
df2=pd.DataFrame({"Actual Y_Test":y_test,"Prediction Data":y_pred})
print("prediction status")
print(df2.to_string())

accuracy2=accuracy_score(y_pred1,y_test)*100
print(accuracy2)

# Boosting the r2 score
model2=XGBRegressor()
model2.fit(X_train,y_train)
y_pred=model2.predict(X_test)
accuracy3=r2_score(y_pred,y_test)*100
print(accuracy3)

# model2 has achieved the best performance, providing the most accurate and reliable predictions among all the models.
