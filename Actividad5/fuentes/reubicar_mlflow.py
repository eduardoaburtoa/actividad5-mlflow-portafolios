"""Reubica solo URIs de artefactos al mover la copia de la base MLflow."""
from pathlib import Path
import sqlite3
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    db=sqlite3.connect(ROOT/'mlflow.db')
    for rid,uri in db.execute('select run_uuid,artifact_uri from runs').fetchall():
        if '/mlartifacts/' in uri:
            suffix=uri.split('/mlartifacts/',1)[1]
            db.execute('update runs set artifact_uri=? where run_uuid=?',((ROOT/'mlartifacts'/suffix).as_uri(),rid))
    db.execute('update experiments set artifact_location=? where name=?',((ROOT/'mlartifacts').as_uri(),'Actividad5_FinancialPhraseBank_v1'))
    db.commit();db.close();print('URIs locales reubicadas; métricas y parámetros intactos.')
