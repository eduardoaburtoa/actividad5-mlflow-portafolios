# Evidencia real de MLflow

Capturas del panel local del 15 de octubre de 2026. Las imágenes no son maquetas.
La base mlflow.db y mlartifacts conservan el seguimiento y los dos modelos.
El CSV exporta 40 corridas: 27 del entrenamiento final (24 candidatos, 2 modelos,
1 comparación) y 13 de un intento anterior que falló al registrar un ejemplo
pandas.Series. Ese fallo se corrigió usando una lista; se conserva su rastro.
La incompatibilidad inicial de SQLAlchemy ocurrió antes de crear una corrida.

Para abrir la evidencia en otro equipo, ejecuta fuentes/reubicar_mlflow.py y
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5055
desde Actividad5. No reentrena ni cambia métricas.
