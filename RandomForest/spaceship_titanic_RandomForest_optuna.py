import matplotlib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import os
import optuna

# Opciones para pandas
pd.set_option('future.no_silent_downcasting', True)

# Paths
path_train = r'../../Dataset/train.csv'
path_test = r'../../Dataset/test.csv'
output_dir = '../Resultados_RandomForest_Optuna'

# Leer datasets
df_train = pd.read_csv(path_train)
df_test = pd.read_csv(path_test)

# Imputación de valores nulos
for col in ["HomePlanet", "CryoSleep", "Destination", "VIP"]:
    df_train[col] = df_train[col].fillna(df_train[col].mode()[0])
    df_test[col] = df_test[col].fillna(df_train[col].mode()[0])

for col in ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]:
    df_train[col] = df_train[col].fillna(0)
    df_test[col] = df_test[col].fillna(0)

df_train["Age"] = df_train["Age"].fillna(df_train["Age"].median())
df_test["Age"] = df_test["Age"].fillna(df_test["Age"].median())
df_train["Name"] = df_train["Name"].fillna("Unknown")
df_test["Name"] = df_test["Name"].fillna("Unknown")

# Feature Engineering para Cabin
df_train[['Deck', 'CabinNumber', 'Side']] = df_train['Cabin'].str.split('/', expand=True)
df_test[['Deck', 'CabinNumber', 'Side']] = df_test['Cabin'].str.split('/', expand=True)

df_train['Deck'] = df_train['Deck'].fillna(df_train['Deck'].mode()[0])
df_test['Deck'] = df_test['Deck'].fillna(df_test['Deck'].mode()[0])
df_train['Side'] = df_train['Side'].fillna(df_train['Side'].mode()[0])
df_test['Side'] = df_test['Side'].fillna(df_test['Side'].mode()[0])

# Conversión segura de CabinNumber
df_train['CabinNumber'] = pd.to_numeric(df_train['CabinNumber'], errors='coerce')
df_test['CabinNumber'] = pd.to_numeric(df_test['CabinNumber'], errors='coerce')

df_train['CabinNumber'] = df_train['CabinNumber'].fillna(df_train['CabinNumber'].median())
df_test['CabinNumber'] = df_test['CabinNumber'].fillna(df_test['CabinNumber'].median())

# Eliminar columnas irrelevantes
df_train.drop(columns=['Cabin', 'Name', 'CabinNumber'], inplace=True)
df_test.drop(columns=['Cabin', 'Name', 'CabinNumber'], inplace=True)

# Codificación de variables categóricas
for col in ["HomePlanet", "Destination"]:
    le = LabelEncoder()
    df_train[col] = le.fit_transform(df_train[col])
    df_test[col] = le.transform(df_test[col])

binary_cols = ["CryoSleep", "VIP", "Transported"]
df_train[binary_cols] = df_train[binary_cols].astype(int)
df_test[binary_cols[:-1]] = df_test[binary_cols[:-1]].astype(int)

deck_mapping = {letter: idx + 1 for idx, letter in enumerate(sorted(df_train["Deck"].dropna().unique()))}
df_train["Deck"] = df_train["Deck"].map(deck_mapping)
df_test["Deck"] = df_test["Deck"].map(deck_mapping)

side_mapping = {'P': 0, 'S': 1}
df_train["Side"] = df_train["Side"].map(side_mapping)
df_test["Side"] = df_test["Side"].map(side_mapping)

# Separar features y target
X = df_train.drop(columns=["Transported", "PassengerId"])
y = df_train['Transported']

# Optuna objective function
def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 100, 500)
    max_depth = trial.suggest_int('max_depth', 3, 15)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    return scores.mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30)

print("\nMejores hiperparámetros encontrados:")
print(study.best_params)

# Entrenamiento final con mejores hiperparámetros
best_params = study.best_params
model = RandomForestClassifier(**best_params, random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

accuracies, precisions, recalls, f1s, specificities = [], [], [], [], []

for train_idx, val_idx in cv.split(X, y):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    conf_matrix = confusion_matrix(y_val, y_pred)
    tn, fp, fn, tp = conf_matrix.ravel()

    accuracies.append(accuracy_score(y_val, y_pred))
    precisions.append(precision_score(y_val, y_pred))
    recalls.append(recall_score(y_val, y_pred))
    f1s.append(f1_score(y_val, y_pred))
    specificities.append(tn / (tn + fp))

print("\nMétricas promedio (5-fold CV) con Optuna:")
print(f"Accuracy: {np.mean(accuracies):.4f}")
print(f"Precisión: {np.mean(precisions):.4f}")
print(f"Recall (Sensibilidad): {np.mean(recalls):.4f}")
print(f"Especificidad: {np.mean(specificities):.4f}")
print(f"F1 Score: {np.mean(f1s):.4f}")

# Predicción sobre test
model.fit(X, y)
X_test = df_test.drop(columns=["PassengerId"])
test_predictions = model.predict(X_test)

submission = pd.DataFrame({
    "PassengerId": df_test["PassengerId"],
    "Transported": test_predictions.astype(bool)
})

os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "submission_rf_14_optuna.csv")
submission.to_csv(output_path, index=False)
print(f"\nArchivo de salida guardado en: {output_path}")
