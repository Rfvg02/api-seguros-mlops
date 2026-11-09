# Entrena el pipeline del taller 1 (preparacion + XGBoost) y lo guarda en pipeline.joblib
# El csv no va en el repo, se descarga del Drive de la clase (medical_insurance.csv)

import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

SEED = 42

seguros = pd.read_csv("medical_insurance.csv")
seguros = seguros.drop_duplicates().reset_index(drop=True)  # 1435 filas repetidas, ver taller 1

X = seguros.drop(columns="charges")
y = seguros["charges"]

num_cols = ["age", "bmi", "children"]
cat_cols = ["sex", "smoker", "region"]

preprocesamiento = ColumnTransformer([
    ("num", Pipeline([("imputar", SimpleImputer(strategy="median")), ("escalar", StandardScaler())]), num_cols),
    ("cat", Pipeline([("imputar", SimpleImputer(strategy="most_frequent")),
                      ("codificar", OneHotEncoder(drop="first", handle_unknown="ignore"))]), cat_cols),
])

pipeline = Pipeline([
    ("preprocesamiento", preprocesamiento),
    ("modelo", XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=SEED)),
])

# primero se mide con train/test como en el taller 1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
pipeline.fit(X_train, y_train)
pred = pipeline.predict(X_test)
print("MAE:", round(mean_absolute_error(y_test, pred), 2))
print("R2 :", round(r2_score(y_test, pred), 4))

# despues se reentrena con todos los datos antes de guardarlo (lo recomendo el profe en la clase 4)
pipeline.fit(X, y)
dump(pipeline, "pipeline.joblib")
print("pipeline guardado en pipeline.joblib")
