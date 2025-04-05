import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# Q.1 => create an array using the range function

arr=np.arange(1,10)
print("Arrya is :-\n",arr ,"\n")

# Q.2 => print the data type

print("Data Type :-\n" ,arr.dtype,"\n")

# Q.3 => create a pandas dataframe from the python dict in car record

dict=[{"id":1,"name":"BMW"},{"id":2,"name":"Audi"}]
d=pd.DataFrame(dict)
print("Old dataframe is :-",d,"\n")

# Q.4 => add one colum in above dataframe 

d['color']=['black','white']
print("New Datafrmae is :-\n" ,d,"\n")


# ******************************************************************************************************

# Q.1 => load the fracture csv file and print the 1st 15 records

df=pd.read_csv("fracture.csv")
print("File is :-",df,"\n")
print("Top 15 records is :-",df.head(15))

# Q.2 => add new coloum named bmi.       formula is :-weight_kg/(height_cm/100)^2

df['bmi']=df['weight_kg']/df['height_cm']**2
print(df)

# Q.3 => split the dataset into test and train

x=df[['weight_kg','height_cm','bmd','bmi']]
y=df['fracture']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
print(x_train,x_test,y_train,y_test)

# Q.4 => build the logistic regression model to predict the considering the age,sex,bmi,bmd

lr=LogisticRegression()
l=LabelEncoder()

df['sex']=l.fit_transform(df['sex'])
df['fracture']=l.fit_transform(df['fracture'])

x=df[['age','sex','bmi','bmd']]
y=df['fracture']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
print(train_test_split)

z=lr.fit(x_train,y_train)
print("Logistic Regression model is :-\n",z ,"\n")


# Q.5 => calculate the accuracy of the model using a confusion matrix

y_pred=lr.predict(x_test)
print("Prediction is :-\n",y_pred)

test=confusion_matrix(y_pred,y_test)
print("Confusion Matrix is :-\n",y_pred,"\n")

# Q.6 => plot the outcomes 

plt.scatter(y_pred,y_test)
plt.show()
