# CHANGELOG y reporte final

## 1.0.0 · 15 de octubre de 2026

- Dataset público AllAgree fijado por revisión; original preservado y cinco
  duplicados retirados. Partición estratificada con semilla20261015.
- Dos modelos entrenados: logística y SVM.24configuraciones y5folds; test reservado.
- SVM seleccionada por CV; F1test0.8635 frente a0.8422 de logística.
- Bootstrap incluye cero; McNemar p0.1892. No se demuestra superioridad concluyente.
- MLflow registra modelos y artefactos; recarga reproduce predicciones.
- Resuelta incompatibilidad SQLAlchemy con2.0.40; ejemplo de modelo convertido de
  Series a lista. Intento fallido preservado en seguimiento.
- Contexto de150clientes simulados y61DICI públicos, sin mezclarlos con entrenamiento.
- Reporte en Word/PDF, notebook Colab, capturas y plan MLOps.

## Decisión

Referencia experimental SVM; revisión humana obligatoria. No se validó
rentabilidad ni producción. Pendientes: español mexicano, licencia comercial,
calibración, evaluación temporal y control de noticias relacionadas.

## 1.1.0 · Aplicación de demostración

Se agrega Streamlit con resultados, inferencia SVM y simulación de costos para clientes ficticios. No cambia el entrenamiento ni las métricas entregadas.

## 1.2.0 · 16 de octubre de 2026 · Auditoría de entrega

- Matriz de cobertura con enlaces directos para los cuatro criterios y cada entregable.
- Capturas de MLflow visibles dentro de la guía de evidencias.
- Visor de corridas, parámetros y métricas reales de SQLite integrado en Streamlit.
- Verificación automática de hashes, parámetros, dataset, 24 candidatos, artefactos y cinco métricas recalculadas con los modelos recargados.
- Corrección de rutas de artefactos al entrenar en otro equipo; notebook actualizado.
- Conservación del código de entrenamiento original para verificar su hash histórico.
- No se reemplazaron resultados originales ni se afirmó un despliegue público de MLflow.
