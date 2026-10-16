"""Audita evidencia MLflow entregada sin entrenar ni alterar la base."""
from pathlib import Path
import hashlib,json,sqlite3
import numpy as np
import pandas as pd
import mlflow.sklearn
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,roc_auc_score
ROOT=Path(__file__).resolve().parents[1]

def verificar():
    expected=pd.read_csv(ROOT/'resultados/comparacion.csv')
    data=pd.read_csv(ROOT/'datos/datos_limp/phrasebank_v1.csv')
    test=data[data.split=='test']
    saved=pd.read_csv(ROOT/'resultados/predicciones_test.csv').set_index('id').loc[test.id]
    manifest=json.loads((ROOT/'datos/manifest.json').read_text())
    source_hashes={hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in ['fuentes/historico/train_20261015.py','fuentes/train.py']}
    report={'estado':'OK','corridas_modelos':[],'comprobaciones':['parámetros y hashes','cinco métricas recalculadas','modelos recargados','predicciones idénticas','24 candidatos enlazados','dataset registrado','artefactos presentes']}
    with sqlite3.connect((ROOT/'mlflow.db').as_uri()+'?mode=ro',uri=True) as db:
        for row in expected.itertuples():
            rid=row.run_id
            status=db.execute('select status from runs where run_uuid=?',(rid,)).fetchone()[0]
            assert status=='FINISHED',rid
            params=dict(db.execute('select key,value from params where run_uuid=?',(rid,)))
            logged=dict(db.execute('select key,value from latest_metrics where run_uuid=?',(rid,)))
            assert params['data_sha256']==manifest['clean_sha256']
            assert params['code_sha256'] in source_hashes
            assert params['folds']=='5' and params['train_rows']=='1807' and params['test_rows']=='452'
            assert db.execute('select count(*) from inputs where destination_id=?',(rid,)).fetchone()[0]>0
            children=db.execute("select run_uuid from tags where key='mlflow.parentRunId' and value=?",(rid,)).fetchall()
            assert len(children)==12
            for (child,) in children:
                assert db.execute('select status from runs where run_uuid=?',(child,)).fetchone()[0]=='FINISHED'
                assert db.execute('select count(*) from params where run_uuid=?',(child,)).fetchone()[0]==3
                assert db.execute('select count(*) from latest_metrics where run_uuid=?',(child,)).fetchone()[0]==2
            folder=ROOT/'mlartifacts'/rid/'artifacts'
            for name in ['model/MLmodel','model/model.pkl','manifest.json',f'grid_{row.model}.csv',f'confusion_{row.model}.png']:
                assert (folder/name).is_file(),name
            model=mlflow.sklearn.load_model(str(folder/'model'))
            pred=model.predict(test.texto)
            assert np.array_equal(pred,saved[row.model].to_numpy())
            prec,rec,f1,_=precision_recall_fscore_support(test.etiqueta,pred,average='macro',zero_division=0)
            scores=model.predict_proba(test.texto) if hasattr(model,'predict_proba') else model.decision_function(test.texto)
            auc=np.mean([roc_auc_score(test.etiqueta==label,scores[:,i]) for i,label in enumerate(model.classes_)])
            metrics=dict(accuracy=accuracy_score(test.etiqueta,pred),precision_macro=prec,recall_macro=rec,f1_macro=f1,roc_auc_ovr_macro=auc)
            for k,v in metrics.items():
                assert np.isclose(v,logged[k],rtol=0,atol=1e-12),k
                assert np.isclose(v,getattr(row,k),rtol=0,atol=1e-12),k
            assert logged['reload_predictions_equal']==1
            report['corridas_modelos'].append({'modelo':row.model,'run_id':rid,'candidatos':len(children),'metricas_verificadas':metrics})
        report['corridas_totales']=db.execute('select count(*) from runs').fetchone()[0]
    return report

if __name__=='__main__':
    report=verificar()
    print(json.dumps(report,ensure_ascii=False,indent=2))
