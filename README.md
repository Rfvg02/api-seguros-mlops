# api-seguros-mlops

Esta es la API del taller 3 de MLOps (EIA). Use el pipeline del taller 1, el de seguros medicos que predice el `charges` con XGBoost, y el `main.py` que vimos en clase ([MLOps-FastAPI](https://github.com/smm-0216/MLOps-FastAPI)), solo le deje el `/predict` y el `/`, el de `/sum` no hacia falta.

El pipeline tiene la preparacion (imputacion, StandardScaler y OneHotEncoder) y el modelo con sus hiperparametros, todo guardado en `pipeline.joblib`. Lo entrena `entrenar_pipeline.py`, pero el csv no esta en el repo porque los datos no se suben, toca bajarlo del Drive de la clase si se quiere volver a entrenar.

El `requirements.txt` tiene las versiones exactas con las que lo entrene, si se instala otra version de scikit-learn el joblib puede no cargar.

## Correrlo en local

Con Python 3.11:

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

y abrir `http://127.0.0.1:8000/docs` para probar el `/predict`. O con curl:

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"data_to_predict\": [[23, \"male\", 34.4, 0, \"no\", \"southwest\"]]}"
```

## Correrlo en EC2 (Ubuntu)

Igual que en clase, con Miniconda:

```bash
git clone https://github.com/Rfvg02/api-seguros-mlops.git
cd api-seguros-mlops
conda create -n fastapi python=3.11
conda activate fastapi
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

Hay que abrir el puerto 8000 en el security group, y queda en `http://<ip-publica>:8000/docs`.

## Dejarla como servicio

Para que no se caiga al cerrar la terminal ni al reiniciar la instancia esta el `fastapi_srv.service`:

```bash
sudo cp fastapi_srv.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl start fastapi_srv
sudo systemctl enable fastapi_srv
sudo systemctl status fastapi_srv
```

Si el entorno o la carpeta quedan en otra ruta toca cambiar el `WorkingDirectory` y el `ExecStart` del .service (se saca con `which python` y se cambia python por uvicorn).
