"""Preparación determinista de Financial PhraseBank sin inferir etiquetas."""
from pathlib import Path
import hashlib, json, re, unicodedata
import pandas as pd
from sklearn.model_selection import train_test_split
ROOT=Path(__file__).resolve().parents[1]
SEED=20261015

def clean_text(text):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC',text)).strip()

def prepare():
    source=ROOT/'datos/datos_ini/Sentences_AllAgree.txt'
    rows=[]
    for i,line in enumerate(source.read_bytes().decode('latin-1').splitlines(),1):
        if not line.strip(): continue
        text,label=line.rsplit('@',1)
        rows.append({'id':f'FPB{i:04d}','texto_original':text,'texto':clean_text(text),'etiqueta':label.strip()})
    df=pd.DataFrame(rows);before=len(df)
    invalid=df.texto.eq('')|~df.etiqueta.isin(['negative','neutral','positive'])
    df=df.loc[~invalid].copy();df['key']=df.texto.str.casefold()
    conflicting=df.groupby('key').etiqueta.nunique();bad=set(conflicting[conflicting>1].index)
    conflicts=int(df.key.isin(bad).sum());df=df.loc[~df.key.isin(bad)]
    dup=int(df.duplicated('key').sum());df=df.drop_duplicates('key').drop(columns='key').reset_index(drop=True)
    train,test=train_test_split(df.index,test_size=.2,stratify=df.etiqueta,random_state=SEED)
    df['split']='train';df.loc[test,'split']='test';df['n_caracteres']=df.texto.str.len()
    dest=ROOT/'datos/datos_limp';dest.mkdir(exist_ok=True)
    df.to_csv(dest/'phrasebank_v1.csv',index=False)
    manifest={'version':'1.0.0','seed':SEED,'original_rows':before,'invalid_rows':int(invalid.sum()),'conflict_rows':conflicts,'duplicate_rows_removed':dup,'clean_rows':len(df),'train_rows':len(train),'test_rows':len(test),'class_counts':df.etiqueta.value_counts().to_dict(),'original_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'clean_sha256':hashlib.sha256((dest/'phrasebank_v1.csv').read_bytes()).hexdigest(),'normalization':'Latin-1; NFKC; whitespace collapse; casefold duplicate key only; no translation or sentiment imputation'}
    (ROOT/'datos/manifest.json').write_text(json.dumps(manifest,indent=2))
    return df,manifest
if __name__=='__main__':
    print(json.dumps(prepare()[1],indent=2))
