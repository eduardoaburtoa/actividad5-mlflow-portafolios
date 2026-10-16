# Revisión de rúbrica y entregables · 16 de octubre de 2026

Revisé la entrega contra los cuatro criterios de 25 puntos de la rúbrica. Esta matriz facilita encontrar evidencia; no sustituye la evaluación del docente ni garantiza una calificación.

| Criterio | Evidencia publicada | Resultado de revisión |
|---|---|---|
| Gestión y preparación de datos (25 %) | [Original](datos/datos_ini/), [limpio](datos/datos_limp/), [diccionario](datos/README.md), [manifiesto](datos/manifest.json), [limpieza](fuentes/datos_prep.py), [gráfico](resultados/calidad_datos.png) | 2,264 originales, 2,259 limpios, cinco duplicados eliminados. Nulos, categorías, normalización y límites documentados. Interpretación en reporte, páginas 4–5. |
| Entrenamiento y ajuste (25 %) | [Entrenamiento](fuentes/train.py), [notebook Colab](fuentes/entrena.ipynb), [rejilla logística](resultados/grid_Logistica.csv), [rejilla SVM](resultados/grid_SVM_lineal.csv), [folds](resultados/cv_folds.csv) | Dos algoritmos, 12 configuraciones por modelo, cinco folds, test reservado y cinco métricas. Justificación técnica en reporte, páginas 6–8. |
| Registro y reproducibilidad MLflow (25 %) | [Capturas y guía](evidencias/README.md), [base](mlflow.db), [artefactos](mlartifacts/), [verificador](fuentes/verificar_mlflow.py), [auditoría ejecutada](evidencias/verificacion_mlflow.json) | 24 candidatos finales + dos modelos + comparación; parámetros, hashes, dataset, firmas y modelos. Recarga y cinco métricas verificadas contra test y SQLite. Reporte, página 9. |
| Comparación e interpretación (25 %) | [Tabla](resultados/comparacion.csv), [gráfico](resultados/comparacion.png), [estadística](resultados/estadistica.json), [reporte](documentos/Actividad5_Eduardo_Aburto.pdf) | Bootstrap pareado y McNemar; ventaja no concluyente; análisis de errores, límites y acciones MLOps. Reporte, páginas 7–11. |

## Lista de entregables

- Datos originales y limpios: publicados con licencia, fuente y versión.
- Programas de preparación y entrenamiento: publicados en `fuentes/`.
- [README](README.md) y [CHANGELOG](CHANGELOG.md): publicados.
- [Notebook](fuentes/entrena.ipynb): publicado; ejecución local CPU documentada; preparado para Colab.
- Evidencia visual: [comparación MLflow](evidencias/mlflow_metricas.png) y [detalle SVM](evidencias/mlflow_svm.png). La consigna admite capturas **o** video: las capturas cubren este entregable.
- [PDF](documentos/Actividad5_Eduardo_Aburto.pdf), [Word](documentos/Actividad5_Eduardo_Aburto.docx) y [Google Docs editable](https://docs.google.com/document/d/10zePpNRuSKNphrLIlZZ3mavDB5lAiLrBx5D90n2A80s/edit): disponibles.
- [Streamlit](app.py): incluye resultados, clasificación, simulación y visor de registros reales de MLflow.

## Corrección de reproducción

La revisión detectó que las rutas absolutas de artefactos conservadas por SQLite podían apuntar al equipo original al iniciar un nuevo entrenamiento. `train.py` ahora reubica la base local antes de usarla; respeta un `MLFLOW_TRACKING_URI` explícito sin modificar un servidor remoto. El notebook también reubica la evidencia antes de verificarla. Se preservó el script original en `fuentes/historico/train_20261015.py` para contrastar el hash registrado en la ejecución entregada. Las métricas originales no se sustituyeron.

El verificador abre SQLite en modo de solo lectura, recarga los modelos con MLflow y recalcula accuracy, precision, recall, F1 y ROC-AUC sobre el test. Contrasta estas cifras con los CSV y las métricas registradas; comprueba además parámetros, vínculos de candidatos, dataset y archivos de artefactos.

## Alcance de negocio

Mantengo 150 personajes ficticios con posiciones entre 5 y 20 millones de pesos para representar perfiles de manera didáctica. El clasificador usa noticias públicas en inglés; no mide rendimiento futuro ni autoriza movimientos. El corretaje del 1 % por compra y por venta es un supuesto académico. El beneficio neto y la compatibilidad del perfil deben prevalecer sobre ingresos por rotación. La cobertura de la rúbrica no implica que el prototipo esté validado para producción financiera.

MLflow funciona localmente, como permite la consigna. GitHub publica la base y las capturas; no ejecuta el servidor MLflow. El visor Streamlit lee registros reales, pero no se presenta como el panel oficial. No se requiere contratar alojamiento para entregar esta actividad.
