# ==========================================
# Ejercicio 1 - Árboles de Decisión
# ==========================================
# Este script aplica técnicas de Árbol de Decisión para resolver
# dos problemas diferentes con el dataset housing_train.csv:
# 1. Regresión → Predicción del precio de las viviendas.
# 2. Clasificación → Clasificación de las viviendas en grupos de precio.
# Incluye análisis exploratorio, correlación, limpieza de datos,
# entrenamiento de modelos y evaluación con métricas.
# ==========================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn import tree
 
sns.set(context='notebook')

# ================================
# 1. Cargar datos
# ================================
# Leemos el dataset con información de viviendas.
# Contiene variables numéricas y categóricas, y la variable objetivo es "SalePrice".
housingTrain = pd.read_csv("housing_train.csv", sep=",")

# ================================
# 2. Análisis exploratorio
# ================================
# Identificamos variables categóricas y mostramos sus frecuencias.
categorical_cols = housingTrain.select_dtypes(include='object').columns
for col in categorical_cols:
    print(f"Variable: {col}")
    print(housingTrain[col].value_counts())
    print("\n")

# Eliminamos columnas que aportan poco al modelo o que son altamente correlacionadas.
housingTrain.drop(["Id", "CentralAir", "OverallCond", "GarageCars", "GarageType",  "MoSold", "GarageArea", "LotFrontage", "YearRemodAdd", "BsmtExposure", "BsmtUnfSF", "HeatingQC", "BsmtFinType1", "Exterior2nd", "KitchenQual", "HalfBath", "FullBath", "BsmtFullBath", "LowQualFinSF","YearBuilt", "MSSubClass", "BsmtFinSF2", "MasVnrArea", "LotConfig", "YrSold", "Exterior1st", "MSZoning", "Functional", "OpenPorchSF", "3SsnPorch", "EnclosedPorch", "WoodDeckSF", "GarageYrBlt", "Fireplaces", "2ndFlrSF", "1stFlrSF", "BedroomAbvGr", "BsmtHalfBath", "LandContour", "LotShape", "Alley", "Street", "KitchenAbvGr", "TotRmsAbvGrd", "ScreenPorch", "Neighborhood", "MiscVal", "PoolArea", "LandSlope", "Condition1", "Condition2", "HouseStyle", "BldgType", "HouseStyle", "BldgType", "Utilities", "PoolQC", "BsmtCond", "RoofMatl", "RoofStyle", "MasVnrType", "ExterCond", "BsmtQual", "Foundation", "ExterQual", "HeatingQC", "SaleCondition", "GarageFinish", "GarageQual"], axis=1, inplace=True)
 
# Renombramos la columna objetivo para facilidad de lectura.
housingTrain.rename(columns={"SalePrice": "Precio"}, inplace=True)
 
print(housingTrain.head())
print(housingTrain.info())

# ================================
# 2.1. Correlación entre variables numéricas
# ================================
numeric_cols = housingTrain.select_dtypes(include=['int64', 'float64'])
corr_matrix = numeric_cols.corr()
print(corr_matrix.head())

plt.figure(figsize=(14,8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", cbar=True)
plt.title("Matriz de correlaciones entre variables numéricas", fontsize=14)
plt.show()

# ================================
# 3. REGRESIÓN: Predicción del Precio
# ================================
# Usamos Árboles de Decisión para predecir el valor de la vivienda.

# Definimos X (features) e y (target).
X = housingTrain.drop("Precio", axis=1)
y = housingTrain["Precio"]

# Convertimos variables categóricas a variables dummy.
X = pd.get_dummies(X, drop_first=True)

# Dividimos en entrenamiento y prueba.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definimos el modelo y la búsqueda de hiperparámetros.
reg = DecisionTreeRegressor(random_state=42)
param_grid = {"max_depth": [2, 3, 4, 5, 10, 20]}

# GridSearch para encontrar la mejor profundidad del árbol.
grid_search_reg = GridSearchCV(reg, param_grid=param_grid, cv=3, scoring="r2", return_train_score=True)
grid_search_reg.fit(X_train, y_train)
 
print("\n===== REGRESIÓN =====")
print("Mejores parámetros:", grid_search_reg.best_params_)
print("Mejor R2 (validación cruzada):", grid_search_reg.best_score_)

# Entrenamos el mejor modelo encontrado.
best_reg = grid_search_reg.best_estimator_

# Evaluamos el modelo en train y test.
y_pred_train = best_reg.predict(X_train)
y_pred_test = best_reg.predict(X_test)

print("MSE Train:", mean_squared_error(y_train, y_pred_train))
print("MSE Test:", mean_squared_error(y_test, y_pred_test))
print("R2 Train:", r2_score(y_train, y_pred_train))
print("R2 Test:", r2_score(y_test, y_pred_test))

# Importancia de características.
feat_importances = pd.Series(best_reg.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(12,5))
sns.barplot(x=feat_importances.head(15), y=feat_importances.head(15).index)
plt.title("Importancia de características (Regresión)")
plt.show()

# Visualización del árbol de decisión de regresión.
plt.figure(figsize=(20,10))
tree.plot_tree(best_reg, feature_names=X.columns, filled=True)
plt.show()

# ================================
# 4. CLASIFICACIÓN: Precio barato/medio/caro
# ================================
# Agrupamos los precios en categorías y usamos Árboles de Decisión para clasificarlos.

def categorizar_precio(valor):
    if valor <= 100000:
        return "Grupo1"  # Viviendas baratas
    elif 100001 <= valor <= 500000:
        return "Grupo2"  # Viviendas de precio medio
    else:
        return "Grupo3"  # Viviendas caras

# Creamos variable categórica.
housingTrain["PrecioCategoria"] = housingTrain["Precio"].apply(categorizar_precio)

print("\nDistribución de clases:")
print(housingTrain["PrecioCategoria"].value_counts(normalize=True))

# Nuevos X e y para clasificación.
X_class = housingTrain.drop(["Precio", "PrecioCategoria"], axis=1)
y_class = housingTrain["PrecioCategoria"]

# Variables dummies.
X_class = pd.get_dummies(X_class, drop_first=True)

# División en train-test.
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_class, y_class, test_size=0.2, random_state=42, stratify=y_class)

# Definimos el modelo y la búsqueda de hiperparámetros.
clf = DecisionTreeClassifier(random_state=42)
param_grid_clf = {"criterion": ["gini", "entropy"], "max_depth": [2, 3, 4]}

# GridSearch para encontrar el mejor clasificador.
grid_search_clf = GridSearchCV(clf, param_grid=param_grid_clf, cv=3, scoring="accuracy", return_train_score=True)
grid_search_clf.fit(X_train_c, y_train_c)
 
print("\n===== CLASIFICACIÓN =====")
print("Mejores parámetros:", grid_search_clf.best_params_)
print("Mejor Accuracy (validación cruzada):", grid_search_clf.best_score_)

# Entrenamos el mejor modelo.
best_clf = grid_search_clf.best_estimator_

# Predicciones.
y_pred_train_c = best_clf.predict(X_train_c)
y_pred_test_c = best_clf.predict(X_test_c)

print("Accuracy Train:", accuracy_score(y_train_c, y_pred_train_c))
print("Accuracy Test:", accuracy_score(y_test_c, y_pred_test_c))

# Matriz de confusión.
cm = confusion_matrix(y_test_c, y_pred_test_c, labels=best_clf.classes_)
ConfusionMatrixDisplay(cm, display_labels=best_clf.classes_).plot(cmap="Blues")
plt.title("Matriz de Confusión (Clasificación Precio)")
plt.show()

# Importancia de características para clasificación.
feat_importances_clf = pd.Series(best_clf.feature_importances_, index=X_class.columns).sort_values(ascending=False)
plt.figure(figsize=(12,5))
sns.barplot(x=feat_importances_clf.head(15), y=feat_importances_clf.head(15).index)
plt.title("Importancia de características (Clasificación)")
plt.show()

# Visualización del árbol de decisión para clasificación.
plt.figure(figsize=(20,10))
tree.plot_tree(best_clf, feature_names=X_class.columns, class_names=best_clf.classes_, filled=True)
plt.show()
