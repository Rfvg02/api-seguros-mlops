# API de seguros medicos - Taller 3 MLOps

API con FastAPI que recibe los datos de una persona y predice el costo del seguro medico (`charges`). El modelo es un pipeline de scikit-learn (imputacion, StandardScaler y OneHotEncoder + XGBoost) guardado en `pipeline.joblib`, el mismo del taller 1. El `main.py` sale del repo de clase [MLOps-FastAPI](https://github.com/smm-0216/MLOps-FastAPI).

## Archivos
- `entrenar_pipeline.py`: entrena el pipeline y lo guarda en `pipeline.joblib`
- `main.py`: la API, con `/` y `/predict`
- `pipeline.joblib`: el pipeline ya entrenado
- `requirements.txt`: las dependencias con sus versiones
- `fastapi_srv.service`: el servicio de systemd para EC2

## Correr en local
Python 3.11.

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Abrir `http://127.0.0.1:8000/docs` y probar `/predict`. Ejemplo con curl:

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"data_to_predict\": [[23, \"male\", 34.4, 0, \"no\", \"southwest\"]]}"
```

Para volver a entrenar el pipeline hay que poner `medical_insurance.csv` en esta carpeta (no se sube al repo) y correr `python entrenar_pipeline.py`.

## Correr en EC2 (Ubuntu)
```bash
git clone https://github.com/<usuario>/api-seguros-mlops.git
cd api-seguros-mlops
conda create -n fastapi python=3.11
conda activate fastapi
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

En el security group de la instancia hay que abrir el puerto 8000. La API queda en `http://<ip-publica>:8000/docs`.

## Dejarla como servicio (systemd)
```bash
sudo cp fastapi_srv.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl start fastapi_srv
sudo systemctl enable fastapi_srv
sudo systemctl status fastapi_srv
```

Con eso la API sigue arriba aunque se cierre la sesion SSH y vuelve a arrancar sola si se reinicia la instancia.
