import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Q.1 => create a 3x3 matrix between 0,9

arr=np.arange(0,9).reshape(3,3)
print("Array is :-\n",arr,'\n')
 
# Q.2 => print datatype 

print("Data Type is :-\n",arr.dtype,"\n")


# Q.3 => create a dataframe name mobie 

d=[{"name":"Nokia","year":2000},{"name":"Oppo","year":2012}]
df1=pd.DataFrame(d)
print("old dataframe is :-\n",df1,'\n')

# Q.4 => add new colum 

df1['colors']=['black','white']
print("New dataframe is :-\n",df1,'\n')

# *********************************************************************************

# Q.1 => load the fracture csv file and print the last 5 records

df=pd.read_csv("fracture.csv")
print(df)

# Q.2 => add new colum name bmi.            formula is :-weight_kg/(height_cm/100)^2

df['bmi']=df['weight_kg']/df['height_cm']**2
print(df)

# Q.3 => split the data set into 30:70 

l=LabelEncoder()

df['fracture']=l.fit_transform(df['fracture'])
df['sex']=l.fit_transform(df['sex'])


x=df[['weight_kg','height_cm','bmd','bmi']]
y=df['fracture']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
print(x_train,x_test,y_train,y_test)

# Q.4 => build a svc model for age ,sex,bmi,bmd         

model=SVC(kernel="sigmoid")                                           # OR linear
z=model.fit(x_test,y_test)
print(z)

y_pred=model.predict(x_test)
print(y_pred)

# Q.5 => confusion matrix

a=confusion_matrix(y_test,y_pred)
print(a)

# Q.6 => plot the outcomes 

plt.scatter(y_test,y_pred)
plt.show()



