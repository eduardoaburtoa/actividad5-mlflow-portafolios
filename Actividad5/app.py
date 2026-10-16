"""Vértice AI: demostración académica, sin datos reales ni órdenes de inversión."""
from pathlib import Path
import pickle
import sqlite3
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
RUN = 'b82e892e82b7451e9da7a0b4cda899f5'
st.set_page_config(page_title='Vértice AI · Portafolios', page_icon='📘', layout='wide')
st.logo(str(ROOT/'documentos/vertice_logo.png'))
st.title('Vértice AI · Revisiones de portafolio')
st.caption('Eduardo Aburto Angulo · Actividad 5 · Prototipo académico')
st.info('La prioridad es el beneficio neto del cliente, compatible con su perfil. Esta demostración no genera órdenes ni recomendaciones personalizadas.')

@st.cache_data
def datos():
    return pd.read_csv(ROOT/'resultados/comparacion.csv'), pd.read_csv(ROOT/'contexto/clientes_simulados.csv')

@st.cache_resource
def modelo():
    # Artefacto propio incluido en el repositorio; no se aceptan archivos externos.
    with (ROOT/f'mlartifacts/{RUN}/artifacts/model/model.pkl').open('rb') as f:
        return pickle.load(f)

res, clientes = datos()
a, b, c, d = st.tabs(['Resultados y evidencia', 'Clasificar una noticia', 'Simulación de costos', 'MLflow · Experimentos'])
with a:
    x,y,z=st.columns(3)
    x.metric('Noticias de prueba', '452')
    y.metric('F1 macro · SVM', '0.8635')
    z.metric('Recall negativo', '0.7541')
    st.subheader('Dos modelos entrenados y registrados')
    st.dataframe(res[['model','accuracy','precision_macro','recall_macro','f1_macro','roc_auc_ovr_macro']],hide_index=True)
    st.image(str(ROOT/'resultados/comparacion.png'))
    st.write('Seleccioné SVM por validación cruzada. La ventaja en test no es concluyente: IC 95 % de la diferencia F1 [−0.00579, 0.04772]; McNemar p = 0.1892.')
    with st.expander('Ver la evidencia auténtica de MLflow'):
        st.image(str(ROOT/'evidencias/mlflow_metricas.png'))
        st.caption('Ejecución local en CPU. El repositorio conserva la base MLflow y los modelos.')
    st.download_button('Descargar reporte PDF', (ROOT/'documentos/Actividad5_Eduardo_Aburto.pdf').read_bytes(), 'Actividad5_Eduardo_Aburto.pdf', 'application/pdf')
with b:
    st.subheader('Sentimiento financiero, para revisión humana')
    st.write('Modelo entrenado con Financial PhraseBank en inglés. Introduce una noticia pública en inglés; no incluyas información de clientes. El resultado no predice rentabilidad ni selecciona fondos.')
    texto=st.text_area('Noticia en inglés', 'The company reported a strong increase in operating profit.', max_chars=4000)
    if st.button('Clasificar noticia',type='primary'):
        if not texto.strip():
            st.warning('Escribe una noticia antes de clasificar.')
        else:
            pred=modelo().predict([texto.strip()])[0]
            etiqueta={'positive':'Positivo','neutral':'Neutral','negative':'Negativo'}[pred]
            st.success(f'Sentimiento estimado: {etiqueta}')
            st.caption('Sin probabilidad de confianza: los márgenes SVM no están calibrados. No se ha validado su desempeño en español.')
with c:
    st.subheader('¿El supuesto beneficio compensa el costo?')
    st.write('150 personajes ficticios, con posiciones de 5 a 20 millones de pesos. Su uso es didáctico: permite ilustrar perfiles sin datos personales reales. Esta simulación no utiliza el clasificador de noticias.')
    cid=st.selectbox('Cliente ficticio',clientes.cliente_id,format_func=lambda v: f"{v} · {clientes.set_index('cliente_id').loc[v,'nombre']}")
    row=clientes.set_index('cliente_id').loc[cid]
    st.write(f"Perfil: **{row.perfil_riesgo}** · Posición simulada: **${row.posicion_mxn:,.0f} MXN** · Fondo histórico: **{row.fondo_actual}**")
    st.caption('El fondo y la posición son datos de contexto heredados; no prueban idoneidad ni vigencia de una recomendación.')
    porcentaje=st.slider('Porcentaje de la posición a analizar',1,100,10)
    diferencial=st.number_input('Diferencial histórico supuesto (puntos porcentuales)',min_value=-50.0,max_value=50.0,value=3.0,step=.25)
    persistencia=st.slider('Persistencia supuesta del diferencial (%)',0,100,35)
    compatible=st.checkbox('En este escenario supongo compatibles el perfil, el horizonte y la liquidez',value=False)
    monto=float(row.posicion_mxn)*porcentaje/100
    costo=monto*.02
    beneficio=monto*diferencial/100*persistencia/100
    neto=beneficio-costo
    x,y,z=st.columns(3)
    x.metric('Monto analizado',f'${monto:,.0f}')
    y.metric('Costo supuesto de ida y vuelta',f'${costo:,.0f}')
    z.metric('Diferencia neta del escenario',f'${neto:,.0f}')
    st.caption('1 % por venta + 1 % por compra sobre la misma base: supuesto didáctico, no tarifa general BBVA verificada. No incluye impuestos ni otros costos. No se resta otra vez un fee ya incorporado en rendimientos netos.')
    if not compatible:
        st.warning('Escenario incompleto: falta revisar compatibilidad. No avanzar a una propuesta de movimiento.')
    elif neto<=0:
        st.warning('Bajo estos supuestos el beneficio no cubre el costo. No se justifica el movimiento por este cálculo.')
    else:
        st.info('El escenario arroja una diferencia positiva; requiere validar los supuestos, las fichas vigentes y la idoneidad con el asesor. No constituye una señal de compra o venta.')
    st.download_button('Descargar clientes ficticios',clientes.to_csv(index=False).encode('utf-8'), 'clientes_ficticios.csv','text/csv')
st.divider()
st.caption('Financial PhraseBank: CC BY-NC-SA 3.0. Uso académico. Las marcas pertenecen a sus titulares; esta demostración no es un producto oficial de BBVA.')

with d:
    st.subheader('Trazabilidad real de MLflow')
    st.write('Consulto la base MLflow incluida en la entrega, en modo de solo lectura. Este visor no sustituye el panel oficial: permite revisar sus registros desde la demostración.')
    with sqlite3.connect((ROOT/'mlflow.db').as_uri()+'?mode=ro', uri=True) as db:
        runs = pd.read_sql_query('SELECT run_uuid AS run_id, name AS nombre, status AS estado FROM runs ORDER BY start_time DESC', db)
        st.dataframe(runs, hide_index=True)
        rid = st.selectbox('Corrida MLflow', runs.run_id, format_func=lambda v: f"{runs.set_index('run_id').loc[v,'nombre']} · {v}")
        st.write('Parámetros registrados')
        st.dataframe(pd.read_sql_query('SELECT key AS parametro,value FROM params WHERE run_uuid=?',db,params=[rid]),hide_index=True)
        st.write('Métricas registradas')
        st.dataframe(pd.read_sql_query('SELECT key AS metrica,value FROM latest_metrics WHERE run_uuid=?',db,params=[rid]),hide_index=True)
    st.caption('27 corridas finales; 13 registros anteriores conservados, incluido un intento fallido. Los IDs permiten distinguirlos.')
    st.download_button('Descargar exportación de MLflow', (ROOT/'evidencias/mlflow_runs.csv').read_bytes(), 'mlflow_runs.csv', 'text/csv')
    st.image(str(ROOT/'evidencias/mlflow_svm.png'), caption='Captura auténtica del panel oficial MLflow: modelo SVM.')
    st.code('cd Actividad5\npython fuentes/reubicar_mlflow.py\npython fuentes/verificar_mlflow.py\nmlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5055',language='bash')
    st.caption('Abre http://127.0.0.1:5055 en el equipo donde ejecutes los comandos. GitHub almacena la evidencia; no hospeda el servidor MLflow.')
