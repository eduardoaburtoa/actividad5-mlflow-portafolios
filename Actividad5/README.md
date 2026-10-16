# Actividad 5 · Preparación de datos y entrenamiento con MLflow

**Eduardo Aburto Angulo · Universidad Tecmilenio · 15 de octubre de 2026**

Doy continuidad al asistente de preparación de revisiones de portafolio para banca
patrimonial, privada y corredores de bolsa. Entreno dos clasificadores sobre un
**dataset público financiero en inglés**. La prioridad del caso es el beneficio
neto del cliente compatible con su perfil. El modelo ordena noticias para revisión;
no decide rotaciones ni pronostica rendimientos.

[Cuaderno Colab](https://colab.research.google.com/github/eduardoaburtoa/actividad5-mlflow-portafolios/blob/main/Actividad5/fuentes/entrena.ipynb) ·
[Reporte Word](documentos/Actividad5_Eduardo_Aburto.docx) · [Reporte PDF](documentos/Actividad5_Eduardo_Aburto.pdf) ·
[Resultados](resultados/comparacion.csv) · [Evidencia MLflow](evidencias/README.md)

## Reproducción

Desde la raíz del repositorio:

```bash
cd Actividad5
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python fuentes/verify.py
```

Para **consultar la corrida entregada**, sin reentrenar:

```bash
python fuentes/reubicar_mlflow.py
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5055
```

Abrir http://127.0.0.1:5055. La reubicación solo ajusta rutas de artefactos a la
ubicación del clon. La base de datos, los modelos y los CSV son evidencia real.
Carga únicamente modelos serializados de una fuente de confianza.

Para **una nueva corrida**, preferiblemente en una copia del repositorio:

```bash
python fuentes/train.py
```

El script sobrescribe los CSV y gráficos de resultados con la nueva ejecución y
crea nuevos IDs en MLflow. Conserva el commit de entrega para consultar la original.
No necesita GPU. Ejecución entregada: CPU local, Python 3.12; no se afirma que fue
realizada en Colab. El notebook instala dependencias, verifica datos, entrena,
compara, abre el panel de MLflow y permite descargar el paquete.

## Datos y prevención de fuga

Financial PhraseBank AllAgree: 2,264 oraciones originales; cinco duplicados
retirados; 2,259 finales. Train: 1,807; test: 452, estratificados con semilla
20261015. 1,386 neutrales, 570 positivas y 303 negativas. [Diccionario](datos/README.md).
Original Latin-1; salida UTF-8; NFKC y espacios. No se traducen noticias ni se
imputan etiquetas. TF-IDF se ajusta dentro de cada fold. Vocabulario máximo 25,000,
frecuencia sublineal y normalización L2. No se incorporan IDs ni split como entradas.
Deduplicación exacta no equivale a separar empresas o noticias relacionadas.

## Modelos y búsqueda

Regresión logística y SVM lineal. GridSearchCV con cinco folds estratificados,
F1 macro como criterio y 12 configuraciones por modelo: C={0.1,1,10},
class_weight={None,balanced}, ngram_range={(1,1),(1,2)}. Total: 120 ajustes CV,
más dos refits finales. No se modifica la rejilla después de mirar el test.

| Modelo | Accuracy | Precision macro | Recall macro | F1 macro | ROC-AUC macro OvR |
|---|---:|---:|---:|---:|---:|
| Logística | 0.8827 | 0.8458 | 0.8387 | 0.8422 | 0.9616 |
| SVM lineal | 0.8982 | 0.8895 | 0.8429 | 0.8635 | 0.9639 |

Seleccionada por CV: SVM, F1 medio 0.8633 (desviación 0.0171), C = 10, sin
ponderación y bigramas. Logística: CV 0.8523 (0.0122), C = 10, balanced, unigramas.
AUC es la media de tres AUC binarias OvR; usa probabilidades para logística y
márgenes para SVM. Estos márgenes **no son probabilidades calibradas**.

Bootstrap pareado estratificado, 2,000 remuestras: diferencia F1(SVM−logística)
0.02136, IC 95 % [−0.00579,0.04772]. McNemar exacto p = 0.1892. No hay evidencia
concluyente de superioridad con α = 0.05. IC condicionado al test, no a reentrenamientos.
La referencia mayoritaria obtiene accuracy 0.6128 y F1 macro 0.2533.

La SVM pierde 15 de 61 noticias negativas (recall 0.7541). No habilito señales
automáticas. Estos resultados no son comparables directamente con la Actividad 4,
que usó 400 notas sintéticas en español y otro protocolo.

## MLflow y trazabilidad

- Dataset, semilla, hash de datos y código, parámetros y métricas registrados.
- Modelos serializados, firma y ejemplo de entrada; recarga verificada con
  predicciones idénticas (`reload_predictions_equal=1`).
- Corrida final: 24 candidatos, 2 modelos, 1 comparación. Se conservan 13 registros
  de un intento previo fallido. [Detalle](evidencias/README.md).
- `mlflow.db`, `mlartifacts/`, capturas auténticas y exportación CSV incluidos.
- Requisitos directos fijados y `requirements-lock.txt` del entorno observado.

## Contexto patrimonial

Los 150 clientes de `contexto/clientes_simulados.csv` son superhéroes y villanos
ficticios con posiciones de 5–20 millones MXN. Se reutiliza la cartera de Actividad 4;
no es el dataset de entrenamiento ni se usan datos personales reales.
Las 61 fichas BBVA consultadas el 15/10/2026 se indexan con URL y SHA-256; hay 693
registros de gastos por serie. Son contexto, no etiquetas ni predictores.

Corretaje de 1 % por venta y 1 % por compra es un supuesto didáctico. Un diferencial histórico de 3 pp
con persistencia de 35 % produce 1.05 pp antes de 2 pp de costo: neto −0.95 pp; no justifica
la operación. No duplicar fees incluidos en rendimientos netos. No equiparar
comisiones totales del fondo con ingreso del banco.

## MLOps pendiente

Corpus externo español mexicano, licencia comercial, calibración, agrupación por
empresa/noticia, evaluación temporal y monitoreo de deriva. Responsables propuestos:
ciencia de datos para validación, tecnología para acceso/respaldo y Cumplimiento
para idoneidad. El asesor documenta y autoriza cada revisión. El costo medido es
tiempo local; no se declara ahorro ni ROI realizado.

## Licencias y fuentes

Dataset y derivados: **CC BY-NC-SA 3.0**, Malo et al.; licencia original en datos_ini.
Su uso comercial requiere acuerdo con los autores. Código propio: MIT; las marcas,
plantillas y documentos de terceros conservan sus derechos. Fuentes:
[Financial PhraseBank](https://huggingface.co/datasets/takala/financial_phrasebank),
[MLflow](https://mlflow.org/docs/latest/ml/tracking/),
[scikit-learn](https://scikit-learn.org/stable/modules/grid_search.html),
[BBVA DICI](https://www.bbva.mx/personas/productos/inversion/normativa-de-fondos.html).
Temas 11 y 12 y rúbrica proporcionados por el usuario. ChatGPT/Codex apoyó esta entrega;
las métricas provienen de ejecución real.

## Actualizar las fichas de contexto (opcional)

Instala `requirements-contexto.txt` y ejecuta `python fuentes/fetch_fund_documents.py` seguido de `python fuentes/extract_fees.py`. Revisa la fecha de consulta y cada serie antes de utilizar cifras nuevas. Estos pasos no intervienen en el entrenamiento.

## Aplicación Streamlit

Desde `Actividad5`, ejecutar:

```bash
python -m pip install -r requirements-app.txt
streamlit run app.py
```

La aplicación consulta los resultados reales, clasifica noticias públicas en inglés
y permite explorar costos con los 150 personajes ficticios. No conecta cuentas
bancarias ni genera órdenes. El cálculo de costos es una simulación independiente
del sentimiento. Las pruebas de interfaz se ejecutan con `python fuentes/test_app.py`.

Para Streamlit Community Cloud: repositorio `eduardoaburtoa/actividad5-mlflow-portafolios`,
rama `main`, archivo principal `streamlit_app.py`, Python 3.12. La raíz incluye las
dependencias y el tema azul. Subir a GitHub no equivale a desplegar en Community Cloud.

[Reporte editable en Google Docs](https://docs.google.com/document/d/10zePpNRuSKNphrLIlZZ3mavDB5lAiLrBx5D90n2A80s/edit).

## Revisión de entrega · 16/10/2026

[Consultar matriz completa de rúbrica y entregables](REVISION_RUBRICA.md).
Después de instalar las dependencias, `python fuentes/verificar_mlflow.py` comprueba la evidencia MLflow, recarga ambos modelos y recalcula las cinco métricas del test. Las capturas están visibles en [la guía de evidencias](evidencias/README.md). Streamlit incluye un visor de corridas, parámetros y métricas leído directamente de SQLite.

El entrenamiento local reubica automáticamente las rutas de artefactos al cambiar de equipo. La evidencia del 15/10/2026 se vincula con el código original preservado en `fuentes/historico/train_20261015.py`; el código actual añade la corrección de portabilidad. No se atribuyen los resultados históricos al código modificado.
