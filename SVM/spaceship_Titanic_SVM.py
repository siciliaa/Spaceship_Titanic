import matplotlib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from scipy.stats import chi2_contingency
import os
from xgboost import XGBClassifier


path_train = r'../../Dataset/train.csv'
path_test = r'../../Dataset/test.csv'
output_dir = '../Resultados_DecisionTree'
# Leemos el dataset:
df_train =pd.read_csv(path_train)
df_test = pd.read_csv(path_test)

print(df_train.info())
print(df_train.isnull().sum())


# Observados que hay valores nulos en casi todas las columnas menos en el id
# y la etiqueta, procedemos a llenar valores usanod moda, mediana, ceros.

for col in ["HomePlanet", "CryoSleep", "Destination", "VIP"]:
    df_train[col] = df_train[col].fillna(df_train[col].mode()[0])
    df_test[col] = df_test[col].fillna(df_train[col].mode()[0])

for col in ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]:
    df_train[col] = df_train[col].fillna(0)
    df_test[col] = df_test[col].fillna(0)


# Rellenar valores nulos en Age con la mediana
df_train["Age"] = df_train["Age"].fillna(df_train["Age"].median())
df_test["Age"] = df_test["Age"].fillna(df_test["Age"].median())

# Rellenar valores nulos en Name con "Unknown"
df_train["Name"] = df_train["Name"].fillna("Unknown")
df_test["Name"] = df_test["Name"].fillna("Unknown")

# Dividir Cabin en Deck, CabinNumber y Side
df_train[['Deck', 'CabinNumber', 'Side']] = df_train['Cabin'].str.split('/', expand=True)
df_test[['Deck', 'CabinNumber', 'Side']] = df_test['Cabin'].str.split('/', expand=True)

# Llenar valores nulos en Deck y Side con la moda (valor más frecuente)
df_train['Deck'] = df_train['Deck'].fillna(df_train['Deck'].mode()[0])
df_test['Deck'] = df_test['Deck'].fillna(df_test['Deck'].mode()[0])

df_train['Side'] = df_train['Side'].fillna(df_train['Side'].mode()[0])
df_test['Side'] = df_test['Side'].fillna(df_test['Side'].mode()[0])

# Convertir CabinNumber a float y rellenar con la mediana
df_train['CabinNumber'] = df_train['CabinNumber'].astype(float)
df_test['CabinNumber'] = df_test['CabinNumber'].astype(float)

df_train['CabinNumber'] = df_train['CabinNumber'].fillna(df_train['CabinNumber'].median())
df_test['CabinNumber'] = df_test['CabinNumber'].fillna(df_test['CabinNumber'].median())

# Eliminar la columna original Cabin correctamente
df_train.drop(columns=['Cabin'], inplace=True)
df_test.drop(columns=['Cabin'], inplace=True)


# Eliminados los valores nulos, vamos a tratar las columnas con datos categóricos:

for col in ["HomePlanet", "Destination"]:
    le = LabelEncoder()
    df_train[col] = le.fit_transform(df_train[col])
    df_test[col] = le.fit_transform(df_test[col])

binary_cols = ["CryoSleep", "VIP", "Transported"]
df_train[binary_cols] = df_train[binary_cols].astype(int)
df_test[binary_cols[:-1]] = df_test[binary_cols[:-1]].astype(int)  # `Transported` no está en test

deck_mapping = {letter: idx + 1 for idx, letter in enumerate(sorted(df_train["Deck"].dropna().unique()))}
df_train["Deck"] = df_train["Deck"].map(deck_mapping)
df_test["Deck"] = df_test["Deck"].map(deck_mapping)

side_mapping = {'P': 0, 'S': 1}
df_train["Side"] = df_train["Side"].map(side_mapping)
df_test["Side"] = df_test["Side"].map(side_mapping)


# Dada la variedad de nombres y apellidos que hay, no se considera que tenga impacto en el modelo, por lo que se elimina.

df_train.drop(columns="Name", inplace=True)
df_test.drop(columns="Name", inplace=True)
df_train.drop(columns="CabinNumber", inplace=True)
df_test.drop(columns="CabinNumber", inplace=True)

print(df_train.dtypes)

# Vamos a estudiar la correlación de las variables:

"""
correlation_matrix = df_train.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.show()
"""

# Vamos a entrenar el modelo:

X_train = df_train.drop(columns=["Transported", "PassengerId"])
y_train = df_train['Transported']

X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42)

model = SVC(kernel="rbf", C=3, probability=True, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)
conf_matrix = confusion_matrix(y_val, y_pred)
tn, fp, fn, tp = conf_matrix.ravel()
specificity = tn / (tn + fp)

print(f"\nMétricas de Evaluación:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Recall (Sensibilidad): {recall:.4f}")
print(f"Especificidad: {specificity:.4f}")
print(f"Precisión: {precision:.4f}")
print(f"F1 Score: {f1:.4f}")


X_test = df_test.drop(columns=["PassengerId"])  # Excluir el ID
test_predictions = model.predict(X_test)

submission = pd.DataFrame({
    "PassengerId": df_test["PassengerId"],  # Mantener el ID original
    "Transported": test_predictions.astype(bool)  # Convertir 1/0 a True/False
})

output_path = os.path.join(output_dir, "submission_10_SVM_EliminandoCabin_tratandoDatos_13depth.csv")
submission.to_csv(output_path, index=False)
