"""Entrena dos pipelines, ajusta con CV y registra evidencia real en MLflow."""
import os
os.environ.setdefault('MPLBACKEND','Agg')
from pathlib import Path
import json,time,platform,hashlib
import numpy as np,pandas as pd,matplotlib.pyplot as plt
import mlflow,mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV,StratifiedKFold,cross_val_score
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,roc_auc_score,confusion_matrix,ConfusionMatrixDisplay,classification_report,f1_score
from sklearn.dummy import DummyClassifier
from scipy.stats import binomtest
from datos_prep import ROOT,SEED,prepare

LABELS=['negative','neutral','positive']
def metrics(y,p,scores):
    pma,rma,fma,_=precision_recall_fscore_support(y,p,average='macro',zero_division=0)
    auc=np.mean([roc_auc_score(np.asarray(y)==c,scores[:,i]) for i,c in enumerate(LABELS)])
    return dict(accuracy=accuracy_score(y,p),precision_macro=pma,recall_macro=rma,f1_macro=fma,roc_auc_ovr_macro=auc)

def main():
    if not os.environ.get('MLFLOW_TRACKING_URI'):
        from reubicar_mlflow import reubicar
        reubicar()
    df,manifest=prepare();out=ROOT/'resultados';out.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(os.environ.get('MLFLOW_TRACKING_URI',f'sqlite:///{ROOT / "mlflow.db"}'))
    experiment='Actividad5_FinancialPhraseBank_v1'
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment,artifact_location=(ROOT/'mlartifacts').as_uri())
    mlflow.set_experiment(experiment)
    train=df[df.split=='train'];test=df[df.split=='test'];X=train.texto;y=train.etiqueta
    folds=list(StratifiedKFold(5,shuffle=True,random_state=SEED).split(X,y))
    fold_map=[]
    for i,(_,val) in enumerate(folds):
        fold_map.extend({'id':train.iloc[j].id,'fold':i} for j in val)
    pd.DataFrame(fold_map).to_csv(out/'cv_folds.csv',index=False)
    records=[];preds={};run_ids=[]
    candidates={'Logistica':LogisticRegression(max_iter=3000,random_state=SEED),'SVM_lineal':LinearSVC(random_state=SEED,dual='auto',max_iter=10000)}
    for name,clf in candidates.items():
        pipe=Pipeline([('tfidf',TfidfVectorizer(lowercase=True,sublinear_tf=True,norm='l2',max_features=25000)),('model',clf)])
        grid={'tfidf__ngram_range':[(1,1),(1,2)],'model__C':[.1,1,10],'model__class_weight':[None,'balanced']}
        with mlflow.start_run(run_name=name) as run:
            mlflow.set_tags({'dataset':'FinancialPhraseBank AllAgree','dataset_version':'1.0.0','seed':SEED,'scope':'academic; English financial news; no trading orders'})
            mlflow.log_params({'seed':SEED,'folds':5,'selection':'f1_macro','train_rows':len(train),'test_rows':len(test),'data_sha256':manifest['clean_sha256'],'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
            mlflow.log_input(mlflow.data.from_pandas(train[['texto','etiqueta']],source='FinancialPhraseBank AllAgree',targets='etiqueta',name='train_v1'),context='training')
            start=time.perf_counter();search=GridSearchCV(pipe,grid,scoring='f1_macro',cv=folds,n_jobs=1,return_train_score=True,error_score='raise').fit(X,y);elapsed=time.perf_counter()-start
            pd.DataFrame(search.cv_results_).to_csv(out/f'grid_{name}.csv',index=False)
            for i,params in enumerate(search.cv_results_['params']):
                with mlflow.start_run(run_name=f'{name}_grid_{i:02d}',nested=True):
                    mlflow.log_params({k:str(v) for k,v in params.items()});mlflow.log_metrics({'cv_f1_mean':search.cv_results_['mean_test_score'][i],'cv_f1_std':search.cv_results_['std_test_score'][i]})
            best=search.best_estimator_;pred=best.predict(test.texto);score=best.predict_proba(test.texto) if name=='Logistica' else best.decision_function(test.texto)
            res=metrics(test.etiqueta,pred,score);res.update(cv_f1_mean=search.best_score_,cv_f1_std=search.cv_results_['std_test_score'][search.best_index_],fit_seconds=elapsed)
            mlflow.log_params({k:str(v) for k,v in search.best_params_.items()});mlflow.log_metrics(res)
            mlflow.sklearn.log_model(best,'model',signature=infer_signature(X.head(3),best.predict(X.head(3))),input_example=X.head(3).tolist())
            uri=f'runs:/{run.info.run_id}/model';loaded=mlflow.sklearn.load_model(uri);assert np.array_equal(pred,loaded.predict(test.texto))
            mlflow.log_metric('reload_predictions_equal',1)
            pd.DataFrame(classification_report(test.etiqueta,pred,output_dict=True)).T.to_csv(out/f'clases_{name}.csv')
            fig,ax=plt.subplots(figsize=(5,4));ConfusionMatrixDisplay.from_predictions(test.etiqueta,pred,labels=LABELS,ax=ax,colorbar=False,cmap='Blues');ax.set_title(name);fig.tight_layout();fig.savefig(out/f'confusion_{name}.png',dpi=130);plt.close(fig)
            mlflow.log_artifact(str(out/f'grid_{name}.csv'));mlflow.log_artifact(str(out/f'confusion_{name}.png'));mlflow.log_artifact(str(ROOT/'datos/manifest.json'))
            record={'model':name,'run_id':run.info.run_id,**res,'best_params':search.best_params_};records.append(record);preds[name]=pred;run_ids.append(run.info.run_id)
    result=pd.DataFrame(records);result.to_csv(out/'comparacion.csv',index=False)
    final=test[['id','etiqueta']].copy()
    for name,pred in preds.items():final[name]=pred
    final.to_csv(out/'predicciones_test.csv',index=False)
    # Comparación pareada sobre el mismo test; modelo elegido SOLO por CV.
    selected=max(records,key=lambda d:d['cv_f1_mean'])['model']
    yt=test.etiqueta.to_numpy();a=preds['Logistica'];b=preds['SVM_lineal'];rng=np.random.default_rng(SEED);deltas=[]
    for _ in range(2000):
        ids=np.concatenate([rng.choice(np.where(yt==c)[0],size=sum(yt==c),replace=True) for c in LABELS])
        deltas.append(f1_score(yt[ids],b[ids],average='macro')-f1_score(yt[ids],a[ids],average='macro'))
    n01=int(((a==yt)&(b!=yt)).sum());n10=int(((a!=yt)&(b==yt)).sum())
    dummy=DummyClassifier(strategy='most_frequent').fit(np.zeros((len(y),1)),y).predict(np.zeros((len(yt),1)))
    stats={'selected_by_cv':selected,'delta_f1_svm_minus_logistica':float(f1_score(yt,b,average='macro')-f1_score(yt,a,average='macro')),'bootstrap_paired_stratified_95_ci':np.quantile(deltas,[.025,.975]).tolist(),'bootstrap_repeats':2000,'mcnemar_exact_p':float(binomtest(n01,n01+n10,.5).pvalue) if n01+n10 else 1.,'discordant_logistic_only_correct':n01,'discordant_svm_only_correct':n10,'dummy_accuracy':accuracy_score(yt,dummy),'dummy_f1_macro':f1_score(yt,dummy,average='macro'),'caveat':'IC condicional al test; no mide incertidumbre de reentrenamiento ni deriva. CV usada para ajustar, no estimación final no sesgada.'}
    (out/'estadistica.json').write_text(json.dumps(stats,indent=2))
    env={'python':platform.python_version(),'platform':platform.platform(),'mlflow':mlflow.__version__,'numpy':np.__version__,'pandas':pd.__version__,'seed':SEED,'execution':'local CPU; notebook Colab provided separately','run_ids':run_ids}
    (out/'entorno.json').write_text(json.dumps(env,indent=2))
    fig,axs=plt.subplots(1,2,figsize=(10,4));df.etiqueta.value_counts().plot.bar(ax=axs[0],color='#004481',rot=0);axs[0].set_title('Distribución de etiquetas');df.boxplot(column='n_caracteres',by='etiqueta',ax=axs[1]);axs[1].set_title('Longitud del texto');fig.suptitle('Financial PhraseBank limpio');fig.tight_layout();fig.savefig(out/'calidad_datos.png',dpi=130);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4));result.set_index('model')[['accuracy','f1_macro','roc_auc_ovr_macro']].plot.bar(ax=ax,color=['#004481','#1464A5','#49A5E6'],rot=0);ax.set_ylim(0,1);ax.set_title('Test reservado sin ajuste');fig.tight_layout();fig.savefig(out/'comparacion.png',dpi=130);plt.close(fig)
    # Registra resultados comparativos como una corrida separada, sin fingir otro modelo.
    with mlflow.start_run(run_name='Comparacion_y_auditoria'):
        mlflow.log_params({'selected_by_cv':selected,'bootstrap_repeats':2000});mlflow.log_metrics({'mcnemar_exact_p':stats['mcnemar_exact_p'],'dummy_f1_macro':stats['dummy_f1_macro']});mlflow.log_artifacts(str(out),'resultados')
    print(result.to_string(index=False));print(json.dumps(stats,indent=2))
if __name__=='__main__':main()
