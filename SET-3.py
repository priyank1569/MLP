# SET-03 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

# Section A

# 1. Create a NumPy array with elements that are multiples of 5
array = np.arange(5, 55, 5)
print(array)

# 2. Print the standard deviation of the array
sd = np.std(array)
print(sd)

# 3. Create a pandas dataframe for Book details
book = {
    'Title': ['Python', 'My Book', 'AI'],
    'Author': ['Sahil', 'Dhruv', 'SK'],
    'Price': [250, 500, 300]
}
df = pd.DataFrame(book)
print(df)

# 4. Rename one column
df.rename(columns={'Price': 'Cost'}, inplace=True)
print(df)

#Section B

# 1. Load the fracture.csv into DataFrame
df = pd.read_csv("fracture.csv")
print(df.head())

# 2. Add a new column 'bmi'
df['bmi'] = df['weight_kg'] / ((df['height_cm'] / 100) ** 2)

# Encode categorical variables
df['sex'] = df['sex'].map({'M': 1, 'F': 0})
df['fracture'] = df['fracture'].map({'fracture': 1, 'no fracture': 0})

# 3. Split the data into 80% train, 20% test
X = df[['age', 'sex', 'bmi', 'bmd']]
y = df['fracture']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Build SVM model
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

# 5. Predict and plot
y_pred_train = svm.predict(X_train)
y_pred_test = svm.predict(X_test)

plt.scatter(y_train, y_pred_train)

plt.scatter(y_test, y_pred_test)

plt.show()

# 6. Confusion matrix & accuracy

a = accuracy_score(y_test, y_pred_test)
print(f"\nAccuracy: {a:.2f}")

b = confusion_matrix(y_test, y_pred_test)
print("Confusion Matrix is :-\n",y_pred_test,"\n")
