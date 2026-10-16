"""Verificación de integridad y evidencia sin reentrenar."""
from pathlib import Path
import json,hashlib
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
m=json.loads((ROOT/'datos/manifest.json').read_text())
for p,key in [('datos/datos_ini/Sentences_AllAgree.txt','original_sha256'),('datos/datos_limp/phrasebank_v1.csv','clean_sha256')]:
    assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==m[key]
df=pd.read_csv(ROOT/'datos/datos_limp/phrasebank_v1.csv')
assert len(df)==2259 and not df.texto.str.casefold().duplicated().any()
a=set(df.loc[df.split=='train','id']);b=set(df.loc[df.split=='test','id']);assert not a&b
assert len(a)==1807 and len(b)==452
p=pd.read_csv(ROOT/'resultados/predicciones_test.csv');assert set(p.id)==b
print('Hashes, duplicados, partición y predicciones verificados.')
