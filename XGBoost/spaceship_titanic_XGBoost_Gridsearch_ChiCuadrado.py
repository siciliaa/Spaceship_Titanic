import matplotlib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
import os
from xgboost import XGBClassifier

# Opciones para pandas
pd.set_option('future.no_silent_downcasting', True)

# Paths
path_train = r'../../Dataset/train.csv'
path_test = r'../../Dataset/test.csv'
output_dir = '../Resultados_XGBoost'

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
X_train = df_train.drop(columns=["Transported", "PassengerId"])
y_train = df_train['Transported']

# --- Selección de características con chi-cuadrado ---
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_train)

n = 10
selector = SelectKBest(score_func=chi2, k=n)
X_selected = selector.fit_transform(X_scaled, y_train)

selected_features = X_train.columns[selector.get_support()].tolist()
print(f"\nMejores {n} características seleccionadas por chi²: {selected_features}")

# Filtrar datasets con esas características
X_train_selected = X_train[selected_features]
X_test_selected = df_test[selected_features]

# Split para validación
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train_selected, y_train, test_size=0.2, random_state=42)

# Grid Search para XGBoost
param_grid = {
    'max_depth': [5],
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1]
}

xgb = XGBClassifier(eval_metric='logloss', random_state=42)

grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    scoring='accuracy',
    cv=3,
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X_train_split, y_train_split)
best_model = grid_search.best_estimator_

# Evaluación en validación
y_pred = best_model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)
conf_matrix = confusion_matrix(y_val, y_pred)
tn, fp, fn, tp = conf_matrix.ravel()
specificity = tn / (tn + fp)

print(f"\nMejores hiperparámetros: {grid_search.best_params_}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precisión: {precision:.4f}")
print(f"Recall (Sensibilidad): {recall:.4f}")
print(f"Especificidad: {specificity:.4f}")
print(f"F1 Score: {f1:.4f}")

# Predicción final en test con features seleccionadas
test_predictions = best_model.predict(X_test_selected)

submission = pd.DataFrame({
    "PassengerId": df_test["PassengerId"],
    "Transported": test_predictions.astype(bool)
})

# Guardar archivo de salida
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "submission_xgb_15_gridsearch_chi2.csv")
submission.to_csv(output_path, index=False)
