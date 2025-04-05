# SET-04 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

# Section A

# 1. Create a NumPy array from a list
list = [12, 45, 67, 89, 23, 56]
array = np.array(list)
print(array)

# 2. Print max and min values
print(np.max(np_array))
print(np.min(np_array))

# 3. Create pandas dataframe from Student Record dictionary
student = {
    'Name': ['Dhruv', 'Sahil', 'Neha','Nish','Ghatotkacha'],
    'Age': [21, 21, 21,21,21],
    'Grade': ['C', 'A', 'A', 'B','B']
}
df = pd.DataFrame(student)
print(df)

# 4. Delete one column from the DataFrame
df.drop(columns='Grade', inplace=True)
print(df)

# Section B

# 1. Load fracture.csv and print first 15 records
df = pd.read_csv("fracture.csv")
print(df.head(15))

# 2. Add 'bmi' column
df['bmi'] = df['weight_kg'] / ((df['height_cm'] / 100) ** 2)

# 3. Encode 'sex' and 'fracture' for ML model and Split dataset
df['sex'] = df['sex'].map({'M': 1, 'F': 0})
df['fracture'] = df['fracture'].map({'fracture': 1, 'no fracture': 0})

X = df[['age', 'sex', 'bmi', 'bmd']]
y = df['fracture']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Build logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# 5. Predict and plot outcomes
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

plt.figure(figsize=(12, 5))

plt.scatter(y_train, y_pred_train)

plt.scatter(y_test, y_pred_test)

plt.tight_layout()
plt.show()

# 6. Confusion matrix and accuracy
a = accuracy_score(y_test, y_pred_test)
print(f"\nAccuracy: {a:.2f}")

b = confusion_matrix(y_test, y_pred_test)
print("Confusion Matrix is :-\n",y_pred_test,"\n")



