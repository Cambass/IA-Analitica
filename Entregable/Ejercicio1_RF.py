# ==========================================
# Ejercicio 1 - Random Forest
# ==========================================
# Este script aplica Random Forest sobre el dataset housing_train.csv
# para resolver dos problemas:
# 1. Regresión → Predicción del precio de las viviendas.
# 2. Clasificación → Clasificación de las viviendas en grupos de precio.
# Incluye preprocesamiento, balanceo de clases, entrenamiento de modelos
# y evaluación con métricas.
# ==========================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler

sns.set(style='whitegrid', context='notebook')

# ================================
# 1. Cargar y preparar datos
# ================================
# Reutilizamos el mismo dataset procesado en Ejercicio1_DT.py.
housingTrain = pd.read_csv("housing_train.csv")
housingTrain.rename(columns={"SalePrice": "Precio"}, inplace=True)

# Creamos variable categórica de precios (3 grupos)
def categorizar_precio(valor):
    if valor <= 100000:
        return "Grupo1"
    elif 100001 <= valor <= 500000:
        return "Grupo2"
    else:
        return "Grupo3"

housingTrain["PrecioCategoria"] = housingTrain["Precio"].apply(categorizar_precio)

# ================================
# 2. Random Forest - REGRESIÓN
# ================================
print("===== RANDOM FOREST REGRESIÓN =====")

# Features y target
X_reg = housingTrain.drop(["Precio", "PrecioCategoria"], axis=1)
y_reg = housingTrain["Precio"]

# Dummies
X_reg = pd.get_dummies(X_reg, drop_first=True)

# Train-Test Split
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Modelo y GridSearch
rf_reg = RandomForestRegressor(random_state=42)
param_grid_reg = {"n_estimators": [100, 200], "max_depth": [5, 10, None], "max_features": ["sqrt", "log2"]}

grid_reg = GridSearchCV(rf_reg, param_grid=param_grid_reg, cv=3, scoring="r2", return_train_score=True)
grid_reg.fit(X_train_r, y_train_r)

print("Mejores parámetros:", grid_reg.best_params_)
print("Mejor R2 (validación cruzada):", grid_reg.best_score_)

best_rf_reg = grid_reg.best_estimator_

# Evaluación
pred_train_r = best_rf_reg.predict(X_train_r)
pred_test_r = best_rf_reg.predict(X_test_r)

print("MSE Train:", mean_squared_error(y_train_r, pred_train_r))
print("MSE Test:", mean_squared_error(y_test_r, pred_test_r))
print("R2 Train:", r2_score(y_train_r, pred_train_r))
print("R2 Test:", r2_score(y_test_r, pred_test_r))

# Importancia de características
feat_imp_reg = pd.Series(best_rf_reg.feature_importances_, index=X_reg.columns).sort_values(ascending=False)
plt.figure(figsize=(12,5))
sns.barplot(x=feat_imp_reg.head(15), y=feat_imp_reg.head(15).index)
plt.title("Importancia de características (RF - Regresión)")
plt.show()

# ================================
# 3. Random Forest - CLASIFICACIÓN
# ================================
print("===== RANDOM FOREST CLASIFICACIÓN =====")

# Features y target
X_clf = housingTrain.drop(["Precio", "PrecioCategoria"], axis=1)
y_clf = housingTrain["PrecioCategoria"]

# Undersampling para balancear clases
dersampler = RandomUnderSampler(random_state=42)
X_clf_res, y_clf_res = dersampler.fit_resample(X_clf, y_clf)

# Dummies
X_clf_res = pd.get_dummies(X_clf_res, drop_first=True)

# Train-Test Split
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_clf_res, y_clf_res, test_size=0.2, random_state=42, stratify=y_clf_res)

# Modelo y GridSearch
rf_clf = RandomForestClassifier(random_state=42)
param_grid_clf = {"n_estimators": [100, 200], "max_depth": [5, 10, None], "criterion": ["gini", "entropy"]}

grid_clf = GridSearchCV(rf_clf, param_grid=param_grid_clf, cv=3, scoring="accuracy", return_train_score=True)
grid_clf.fit(X_train_c, y_train_c)

print("Mejores parámetros:", grid_clf.best_params_)
print("Mejor Accuracy (validación cruzada):", grid_clf.best_score_)

best_rf_clf = grid_clf.best_estimator_

# Evaluación
pred_train_c = best_rf_clf.predict(X_train_c)
pred_test_c = best_rf_clf.predict(X_test_c)

print("Accuracy Train:", accuracy_score(y_train_c, pred_train_c))
print("Accuracy Test:", accuracy_score(y_test_c, pred_test_c))
print("\nReporte de clasificación:\n", classification_report(y_test_c, pred_test_c))

# Matriz de confusión
cm = confusion_matrix(y_test_c, pred_test_c, labels=best_rf_clf.classes_)
ConfusionMatrixDisplay(cm, display_labels=best_rf_clf.classes_).plot(cmap="Blues")
plt.title("Matriz de Confusión (RF - Clasificación)")
plt.show()

# Importancia de características
feat_imp_clf = pd.Series(best_rf_clf.feature_importances_, index=X_train_c.columns).sort_values(ascending=False)
plt.figure(figsize=(12,5))
sns.barplot(x=feat_imp_clf.head(15), y=feat_imp_clf.head(15).index)
plt.title("Importancia de características (RF - Clasificación)")
plt.show()

# ==========================================
# Comentarios finales:
# - RF en regresión suele mejorar significativamente los resultados frente a un árbol individual.
# - RF en clasificación es más robusto y menos propenso al overfitting.
# - La importancia de variables como OverallQual y GrLivArea se mantiene en ambos modelos.
# ==========================================
