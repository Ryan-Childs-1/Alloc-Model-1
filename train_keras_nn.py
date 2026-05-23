import os
os.environ.setdefault('KERAS_BACKEND','torch')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import glob, os, json, hashlib, warnings
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import keras
from keras import layers, regularizers
warnings.filterwarnings('ignore')
SEED=42
np.random.seed(SEED)
OUT_DIR='/mnt/data/allocation_ai_keras_final_flat'; os.makedirs(OUT_DIR, exist_ok=True)
FILES=sorted(glob.glob('/mnt/data/*Daily Allocation*.csv'))
NUMERIC_BASE=['Department Id','Class Id','Line Id','Site','Square Footage','Zone','MIL','FLM','DC FLM','Orgs','Cost','Retail','ATG Retail','GM Pct','L30','D30','D60','LW','TTM','Qoh','Supply','Allocated','Intrans','Store Transfer','QTY Reserve','Store PO Qty','Dc Qoh','Dc Avail','DC Staged','DC RV','DC PO QTY','Avg. WOC','MIL.1','FLM.1','Days','Proj. Demand','Alloc. Rec.']
CATEGORICAL_BASE=['Vendor','Vendor Site Id','Brand','Dcl','Class Name','Line Name','Product ID','Pcode Description','Color','Size','Item','Description','Mfg Code','UPC','Status','Status 300','Site Name','State','Region','Buyer Name','Planner Code','Private Label','Season Code','Store Size','Rank','Supply In Stock','New','Store Flag','SKU Flag','Flag']
NEEDED=sorted(set(NUMERIC_BASE+CATEGORICAL_BASE+['Final Alloc.']))
HASH_DIM=128

def to_num(s):
    return pd.to_numeric(s.astype(str).str.replace(',','',regex=False).str.replace('$','',regex=False).str.replace('%','',regex=False).str.replace('(','-',regex=False).str.replace(')','',regex=False).replace({'nan':np.nan,'None':np.nan,'':np.nan,' ':np.nan,'-':np.nan}), errors='coerce')
def get_col(df,c,default=np.nan):
    return df[c] if c in df.columns else pd.Series(default,index=df.index)
def read_signal(path):
    chunks=[]; total=0; pos=0; signal_n=0
    for ch in pd.read_csv(path, header=1, dtype=str, low_memory=False, chunksize=50000, usecols=lambda c: str(c).strip() in set(NEEDED)):
        total += len(ch)
        y=to_num(get_col(ch,'Final Alloc.',np.nan)).fillna(0)
        alloc=to_num(get_col(ch,'Alloc. Rec.',np.nan)).fillna(0)
        flag=get_col(ch,'Flag','').fillna('').astype(str).str.upper()
        sig=flag.str.contains('ALLOCATE|REVIEW',regex=True) | (alloc>0) | (y>0)
        chunks.append(ch.loc[sig].copy())
        pos += int((y>0).sum()); signal_n += int(sig.sum())
    df=pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    df['__source_file']=os.path.basename(path)
    return df, {'file':os.path.basename(path),'rows':int(total),'signal_rows_used':int(len(df)),'positive_rows':int(pos),'all_signal_rows':int(signal_n)}
def derive_numeric(df):
    out=pd.DataFrame(index=df.index)
    for c in NUMERIC_BASE: out[c]=to_num(df[c]) if c in df.columns else np.nan
    flm=out['FLM.1'].fillna(out['FLM']).replace(0,np.nan); mil=out['MIL.1'].fillna(out['MIL'])
    d60=out['D60']; d30=out['D30']; l30=out['L30']; lw=out['LW']; ttm=out['TTM']; supply=out['Supply']; qoh=out['Qoh']; alloc=out['Alloc. Rec.']
    out['effective_flm']=flm; out['effective_mil']=mil
    out['d60_minus_supply']=d60-supply; out['d30_minus_supply']=d30-supply; out['l30_minus_supply']=l30-supply; out['proj_minus_supply']=out['Proj. Demand']-supply
    out['alloc_rec_over_flm']=alloc/flm; out['alloc_rec_minus_gap']=alloc-(d60-supply); out['supply_over_d60']=supply/d60.clip(lower=1); out['qoh_over_l30']=qoh/l30.clip(lower=1)
    out['ttm_monthly']=ttm/12; out['lw_monthly_runrate']=lw*4.29
    out['recent_velocity']=l30.fillna(0)*.35+d30.fillna(0)*.20+(d60.fillna(0)/2)*.25+(lw.fillna(0)*4.29)*.20
    out['velocity_minus_supply']=out['recent_velocity']-supply; out['dc_avail_over_alloc_rec']=out['Dc Avail']/alloc.clip(lower=1); out['dc_avail_over_flm']=out['Dc Avail']/flm.clip(lower=1)
    out['store_pipeline']=out[['Allocated','Intrans','Store Transfer','QTY Reserve','Store PO Qty']].sum(axis=1,min_count=1)
    out['available_store_supply']=qoh+out['store_pipeline'].fillna(0); out['d60_gap_after_pipeline']=d60-out['available_store_supply']
    flag=get_col(df,'Flag','').fillna('').astype(str).str.upper()
    out['is_allocate']=flag.str.fullmatch('ALLOCATE').fillna(False).astype(float); out['is_review']=flag.str.contains('REVIEW',regex=False).fillna(False).astype(float); out['is_z_no_alloc']=flag.str.contains('NO ALLOC',regex=False).fillna(False).astype(float)
    return out.replace([np.inf,-np.inf],np.nan)
def stable_bucket(text, dim=HASH_DIM): return int.from_bytes(hashlib.blake2b(str(text).encode('utf-8',errors='ignore'),digest_size=8).digest(),'little')%dim
def hash_cats(df):
    cats=[c for c in CATEGORICAL_BASE if c in df.columns]
    mat=np.zeros((len(df),HASH_DIM),dtype=np.float32); rows=np.arange(len(df))
    for c in cats:
        vals=get_col(df,c,'__MISSING__').fillna('__MISSING__').astype(str).str.strip().replace('', '__BLANK__').to_numpy()
        uniques, inv=np.unique(vals, return_inverse=True)
        buckets=np.array([stable_bucket(c+'='+u) for u in uniques], dtype=np.int32)
        np.add.at(mat,(rows,buckets[inv]),1.0)
    if cats: mat/=np.sqrt(len(cats))
    return mat
frames=[]; summaries=[]
for f in FILES:
    d,s=read_signal(f); frames.append(d); summaries.append(s); print(s, flush=True)
df=pd.concat(frames, ignore_index=True)
y=to_num(get_col(df,'Final Alloc.',np.nan)).fillna(0).clip(lower=0).to_numpy(np.float32); y_log=np.log1p(y)
num=derive_numeric(df); medians=num.median(numeric_only=True).replace([np.inf,-np.inf],np.nan).fillna(0); scaler=StandardScaler(); x_num=scaler.fit_transform(num.fillna(medians)).astype(np.float32); x_cat=hash_cats(df); X=np.hstack([x_num,x_cat]).astype(np.float32)
idx=np.arange(len(df)); train_idx,val_idx=train_test_split(idx,test_size=.2,random_state=SEED,stratify=(y>0).astype(int))
w=np.where(y>0,5.0,1.0).astype(np.float32)
inp=keras.Input(shape=(X.shape[1],),name='allocation_features')
x=layers.BatchNormalization()(inp)
x=layers.Dense(64,kernel_regularizer=regularizers.l2(2e-5))(x); x=layers.BatchNormalization()(x); x=layers.Activation('gelu')(x); x=layers.Dropout(.12)(x)
r=x; x=layers.Dense(64,kernel_regularizer=regularizers.l2(2e-5))(x); x=layers.BatchNormalization()(x); x=layers.Activation('gelu')(x); x=layers.Dropout(.08)(x); x=layers.Dense(64,kernel_regularizer=regularizers.l2(2e-5))(x); x=layers.Add()([x,r]); x=layers.Activation('gelu')(x)
x=layers.Dense(32)(x); x=layers.Activation('gelu')(x); x=layers.Dropout(.05)(x)
out=layers.Dense(1,activation='linear',name='log_final_alloc')(x)
model=keras.Model(inp,out)
model.compile(optimizer=keras.optimizers.AdamW(learning_rate=9e-4,weight_decay=3e-5), loss=keras.losses.Huber(delta=.8), metrics=['mae'])
h=model.fit(X[train_idx], y_log[train_idx], sample_weight=w[train_idx], validation_data=(X[val_idx], y_log[val_idx], w[val_idx]), epochs=40, batch_size=2048, verbose=2, callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss',patience=9,restore_best_weights=True),keras.callbacks.ReduceLROnPlateau(monitor='val_loss',patience=4,factor=.5,min_lr=2e-5)])
log_pred=model.predict(X[val_idx],batch_size=2048,verbose=0).reshape(-1); raw=np.expm1(np.maximum(log_pred,0))
metrics={'created_at':datetime.utcnow().isoformat()+'Z','model_type':'Keras 3 Torch-backend residual neural network, signal-row allocation regressor','source_files':summaries,'full_source_rows_seen':int(sum(s['rows'] for s in summaries)),'signal_training_rows':int(len(df)),'positive_rows':int((y>0).sum()),'feature_count':int(X.shape[1]),'numeric_feature_count':int(len(num.columns)),'hash_dim':HASH_DIM,'epochs_trained':int(len(h.history['loss'])),'validation_rows':int(len(val_idx)),'validation_positive_rows':int((y[val_idx]>0).sum()),'validation_mae_raw_units':float(mean_absolute_error(y[val_idx],raw)),'validation_rmse_raw_units':float(np.sqrt(mean_squared_error(y[val_idx],raw))),'validation_positive_mae_raw_units':float(mean_absolute_error(y[val_idx][y[val_idx]>0],raw[y[val_idx]>0])),'history_tail':{k:[float(vv) for vv in vals[-10:]] for k,vals in h.history.items()}}
bundle={'version':'allocation_ai_keras_nn_signal_v4_2026_05_22','numeric_base':NUMERIC_BASE,'categorical_base':CATEGORICAL_BASE,'numeric_columns':list(num.columns),'medians':medians,'scaler':scaler,'hash_dim':HASH_DIM,'metrics':metrics}
model.save(os.path.join(OUT_DIR,'allocation_ai_keras_nn_model.keras'))
joblib.dump(bundle, os.path.join(OUT_DIR,'allocation_ai_keras_preprocessor.joblib'))
with open(os.path.join(OUT_DIR,'training_metrics.json'),'w') as f: json.dump(metrics,f,indent=2)
sample=df.iloc[val_idx[:2000]].copy(); sample['AI_val_raw_prediction']=raw[:2000]; sample.to_csv(os.path.join(OUT_DIR,'keras_nn_validation_sample.csv'), index=False)
print(json.dumps(metrics,indent=2))
