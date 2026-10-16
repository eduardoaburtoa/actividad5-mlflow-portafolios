"""Prueba de carga, inferencia, entrada vacía y escenarios de la aplicación."""
from pathlib import Path
from streamlit.testing.v1 import AppTest
root=Path(__file__).resolve().parents[1]
a=AppTest.from_file(str(root/'app.py'),default_timeout=30).run()
assert not a.exception, a.exception
assert len(a.selectbox[0].options)==150
a.button[0].click().run();assert not a.exception
assert 'Sentimiento estimado:' in a.success[0].value
a.text_area[0].set_value('   ');a.button[0].click().run()
assert any('Escribe una noticia' in x.value for x in a.warning)
a.checkbox[0].check().run()
assert any('no cubre el costo' in x.value for x in a.warning)
a.number_input[0].set_value(20.0).run()
assert any('diferencia positiva' in x.value for x in a.info)
assert not a.exception
print('Streamlit: carga, 150 clientes, inferencia, entrada vacía y dos escenarios verificados.')

# El visor lee las 40 corridas de la evidencia original sin modificar SQLite.
import sqlite3
with sqlite3.connect((root/'mlflow.db').as_uri()+'?mode=ro',uri=True) as db:
    expected_runs=db.execute('select count(*) from runs').fetchone()[0]
assert any(len(frame.value)==expected_runs and 'run_id' in frame.value.columns for frame in a.dataframe)
assert any(tab.label=='MLflow · Experimentos' for tab in a.tabs)
print('Visor MLflow: corridas reales visibles y sin errores.')
