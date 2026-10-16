"""Reubica exclusivamente artefactos de la base local entregada."""
from pathlib import Path
import sqlite3
ROOT = Path(__file__).resolve().parents[1]

def reubicar(root=ROOT):
    root = Path(root).resolve()
    if not (root / 'mlflow.db').exists():
        return
    with sqlite3.connect(root / 'mlflow.db') as db:
        for rid, uri in db.execute('select run_uuid,artifact_uri from runs').fetchall():
            if uri and uri.startswith('file:') and '/mlartifacts/' in uri:
                suffix = uri.split('/mlartifacts/', 1)[1]
                target = (root / 'mlartifacts' / suffix).resolve()
                if not target.is_relative_to(root / 'mlartifacts'):
                    raise ValueError('Ruta de artefacto fuera del proyecto')
                db.execute('update runs set artifact_uri=? where run_uuid=?', (target.as_uri(), rid))
        db.execute('update experiments set artifact_location=? where name=?',
                   ((root / 'mlartifacts').as_uri(), 'Actividad5_FinancialPhraseBank_v1'))

if __name__ == '__main__':
    reubicar()
    print('URIs locales reubicadas; métricas y parámetros intactos.')
