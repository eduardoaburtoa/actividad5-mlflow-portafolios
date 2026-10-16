# Actividad 5 · Vértice AI

Eduardo Aburto Angulo · Preparación de datos, entrenamiento y seguimiento con MLflow.

![Vértice AI](Actividad5/documentos/vertice_logo.png)

- [Entrega completa, metodología y reproducción](Actividad5/README.md)
- [Reporte PDF](Actividad5/documentos/Actividad5_Eduardo_Aburto.pdf) · [Word](Actividad5/documentos/Actividad5_Eduardo_Aburto.docx)
- [Google Docs editable](https://docs.google.com/document/d/10zePpNRuSKNphrLIlZZ3mavDB5lAiLrBx5D90n2A80s/edit)
- [Abrir notebook en Colab](https://colab.research.google.com/github/eduardoaburtoa/actividad5-mlflow-portafolios/blob/main/Actividad5/fuentes/entrena.ipynb)
- [Evidencia MLflow](Actividad5/evidencias/README.md)

## Streamlit

```bash
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

Consulta resultados, clasifica noticias en inglés con la SVM entrenada y explora costos con 150 clientes ficticios. No genera órdenes de inversión. Para Community Cloud: rama `main`, archivo `streamlit_app.py`, Python 3.12. Aplicación probada localmente; el despliegue en un servicio de alojamiento es un paso separado.

## Resultado

SVM: F1 macro 0.8635; regresión logística: 0.8422. La ventaja no es estadísticamente concluyente. Entrenamiento real en CPU, 24 configuraciones evaluadas con cinco folds, modelos y evidencias incluidos. Datos financieros públicos con licencia académica no comercial; no contienen datos de clientes reales.

## Revisión de entrega y MLflow

- [Matriz de cobertura de la rúbrica y entregables](Actividad5/REVISION_RUBRICA.md).
- [Capturas del panel oficial y guía MLflow](Actividad5/evidencias/README.md).
- [Auditoría ejecutada de parámetros, métricas y modelos](Actividad5/evidencias/verificacion_mlflow.json).

Streamlit incorpora la pestaña **MLflow · Experimentos**, que consulta los registros reales de la base entregada. Para abrir el panel oficial, sigue la guía de evidencias. Ambos pueden ejecutarse localmente; GitHub conserva los archivos y no hospeda esos servidores.
