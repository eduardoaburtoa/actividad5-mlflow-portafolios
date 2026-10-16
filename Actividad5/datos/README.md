# Diccionario y versión 1.0.0

Fuente: Financial PhraseBank v1.0, Malo et al. (2014), espejo mantenido por takala.
URL y revisión exacta: fuente.json. CC BY-NC-SA3.0; licencia y README originales
acompañan el TXT. AllAgree selecciona acuerdo total; puede sesgar hacia noticias
menos ambiguas. Fuente en inglés, no traducida, de contexto financiero histórico.

| Campo limpio | Tipo | Regla |
|---|---|---|
| id | string | FPB+número de línea del TXT |
| texto_original | string | texto fuente decodificado Latin-1 |
| texto | string | Unicode NFKC y espacios compactados; predictor |
| etiqueta | categoría | negative,neutral,positive; objetivo |
| split | categoría | train/test,80/20 estratificado,semilla20261015 |
| n_caracteres | entero | longitud normalizada; solo exploración |

Nulos/textos vacíos:0; etiquetas inválidas:0; conflictos de etiqueta:0;
duplicados eliminados:5. No se imputa ni altera el objetivo. Se conservan noticias
largas, cifras y negaciones. Caracteres extraños de la fuente no se reconstruyen
sin evidencia. Tokenización y minúsculas las aplica TF-IDF dentro de cada fold;
normalización L2 y frecuencia sublineal. No hay variables categóricas predictoras.

SHA-256 y conteos:manifest.json. Los IDs de partición quedan en el CSV limpio;
los folds de entrenamiento en resultados/cv_folds.csv. Git versiona archivos
pequeños; no se requiere DVC en esta escala. Una limpieza nueva implica versión
y entrada en CHANGELOG; no reemplazar silenciosamente una evidencia publicada.
