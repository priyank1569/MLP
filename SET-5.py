#SET-05 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

# Section A

# 1. Create a 5x5 NumPy array with values from 0 to 24
array = np.arange(25).reshape(5, 5)
print(array)

# 2. Print the size and shape of the array
print(array.size)
print(array.shape)

# 3. Create a pandas dataframe for Mobile phone details
mobiles = {
    'Brand': ['Samsung', 'Apple', 'Xiaomi'],
    'Model': ['M35', 'iPhone 15', 'Redmi'],
    'Price': [100,10,80]
}
df = pd.DataFrame(mobiles)
print(df)

# 4. Add one more column
df['RAM'] = ['8GB', '6GB', '4GB']
print(df)

# Section B

# 1. Load fracture.csv
df = pd.read_csv("fracture.csv")
print(df.tail())

# 2. Add BMI column
df['bmi'] = df['weight_kg'] / ((df['height_cm'] / 100) ** 2)

# 3. Encode categorical data
df['sex'] = df['sex'].map({'M': 1, 'F': 0})
df['fracture'] = df['fracture'].map({'fracture': 1, 'no fracture': 0})

# 4. Prepare data
X = df[['age', 'sex', 'bmi', 'bmd']]
y = df['fracture']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 5. Build KNN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

y_pred_train = knn.predict(X_train)
y_pred_test = knn.predict(X_test)

plt.scatter(y_train, y_pred_train)

plt.scatter(y_test, y_pred_test)

plt.show()

# 6. Confusion matrix & accuracy
a = accuracy_score(y_test, y_pred_test)
print(f"\nAccuracy: {a:.2f}")

b = confusion_matrix(y_test, y_pred_test)
print("Confusion Matrix is :-\n",y_pred_test,"\n")
