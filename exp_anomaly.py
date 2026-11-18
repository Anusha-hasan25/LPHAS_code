import numpy as np, json, time
from lphas_lib import *
from sklearn.ensemble import IsolationForest
log=lambda s:print(s,flush=True)
import sys,os
SEEDS=[int(a) for a in sys.argv[1:]] or [0]
out=json.load(open('results_anomaly.json')) if os.path.exists('results_anomaly.json') else {'seeds':{}}
TR,VA,TE=20000,26000,40000
def shift(ev,off): return [(a-off,b-off,t,c) for a,b,t,c in ev]
def adapt(s,w=400):
    o=np.empty_like(s)
    for i in range(len(s)): o[i]=s[i]/(np.median(s[max(0,i-w):i+1])+1e-6)
    return o
def best_on_val(sv,evv,mode):
    best=None
    if mode=="ndt": grid=[0.0,0.05,0.13]
    else: grid=[0.97,0.98,0.99,0.995]
    for g in grid:
        fl=ndt(sv,p_prune=g) if mode=="ndt" else sv>=np.quantile(sv[:6000//2] if False else sv,g)
        f=event_eval(fl,evv)["f1"]
        if best is None or f>best[0]: best=(f,g)
    return best[1]
def apply(s_val_nom,s_val,s_te,evv,evt,mode):
    g=best_on_val(s_val,evv,mode)
    if mode=="ndt": fl=ndt(s_te,p_prune=g)
    else: fl=s_te>=np.quantile(s_val_nom,g)       # quantile of nominal-val scores
    r=event_eval(fl,evt); r["param"]=g; return r,fl
for sd in SEEDS:
    t0=time.time(); log(f"seed {sd}")
    Xc,_,_,storm=gen_series(TE,sd)
    Xa,lab,ev,_=gen_series(TE,sd,anom_ranges=[(VA-5200,VA-300,12),(VA+800,TE-300,36)])
    mu,sdv=Xc[:TR].mean(0),Xc[:TR].std(0); N=lambda X:(X-mu)/sdv
    Ztr=N(Xc[:TR]); Zvn=N(Xc[TR:VA]); Zva=N(Xa[TR:VA]); Zte=N(Xa[VA:])
    evv=shift([e for e in ev if e[1]<=VA],TR); evt=shift([e for e in ev if e[0]>=VA],VA)
    m=LSTM(32,seed=sd); hist=train(m,Ztr,Zvn,epochs=60,seed=sd,log=log)
    pvn=predict_stream(m,Zvn); esd=np.nanstd(Zvn-pvn,0)
    pv=predict_stream(m,Zva); pt=predict_stream(m,Zte)
    s_vn=score_series(Zvn,pvn,esd); s_v=score_series(Zva,pv,esd); s_t=score_series(Zte,pt,esd)
    res={}
    res["LSTM + NDT"],fl0=apply(s_vn,s_v,s_t,evv,evt,"ndt")
    res["LSTM + fixed threshold"],_=apply(s_vn,s_v,s_t,evv,evt,"fix")
    res["LSTM + adaptive scale + NDT"],fl_ad=apply(adapt(s_vn),adapt(s_v),adapt(s_t),evv,evt,"ndt")
    pp=lambda Z:np.vstack([Z[:1],Z[:-1]]); sdp=np.nanstd(Zvn-pp(Zvn),0)
    res["Persistence + NDT"],_=apply(score_series(Zvn,pp(Zvn),sdp),score_series(Zva,pp(Zva),sdp),score_series(Zte,pp(Zte),sdp),evv,evt,"ndt")
    def feats(Z,k=8): return np.concatenate([np.zeros((k,Z.shape[1]*k)),np.stack([Z[i-k:i].ravel() for i in range(k,len(Z))])])
    iso=IsolationForest(n_estimators=100,random_state=sd).fit(feats(Ztr)[8::5])
    isc=lambda Z:ewma(-iso.score_samples(feats(Z)),10)
    res["Isolation Forest"],_=apply(isc(Zvn),isc(Zva),isc(Zte),evv,evt,"fix")
    for k in res: res[k]["fa_per_1000"]=res[k]["fp"]/(len(Zte)/1000)
    # per-type recall for LSTM + adaptive + NDT
    types={}
    for a,b,t,c in evt:
        hit=any(x<b+10 and y>a-10 for x,y in sequences(fl_ad)); types.setdefault(t,[]).append(hit)
    ptype={t:float(np.mean(v)) for t,v in types.items()}
    # quantisation sweep (adaptive+NDT with val-chosen prune)
    gq=res["LSTM + adaptive scale + NDT"]["param"]; q={}
    for bits in (8,6,4,3):
        mq,size=quantize_weights(m,bits); pq=predict_stream(mq,Zte); sq=adapt(score_series(Zte,pq,esd))
        r=event_eval(ndt(sq,p_prune=gq),evt); r["rmse"]=float(np.sqrt(np.nanmean((Zte-pq)**2))); r["bytes"]=float(size); q[bits]=r
    r=dict(res["LSTM + adaptive scale + NDT"]); r["rmse"]=float(np.sqrt(np.nanmean((Zte-pt)**2))); r["bytes"]=float(m.n_params()*4); q["fp32"]=r
    # autoregressive rollout fidelity on nominal test region
    Zro=N(Xc[VA:]); st_ro=storm[VA:]; ctx,K=32,60; starts=np.arange(200,len(Zro)-K-ctx-2,20)
    seq=np.stack([Zro[s:s+ctx] for s in starts]); cur=m.forward(seq)[:,-1]; preds=[]
    for k in range(K):
        seq=np.concatenate([seq,cur[:,None]],1); cur=m.forward(seq)[:,-1]; preds.append(cur)
    P=np.stack(preds,1); T_=np.stack([Zro[s+ctx:s+ctx+K] for s in starts]); nr=np.sqrt(((P-T_)**2).mean(2))
    sf=np.array([st_ro[s+ctx:s+ctx+K].mean() for s in starts])
    out["seeds"][str(sd)]=dict(hist=hist,res=res,quant=q,ptype=ptype,calm=nr[sf<1.3].mean(0).tolist(),storm=nr[sf>1.8].mean(0).tolist(),
        n_calm=int((sf<1.3).sum()),n_storm=int((sf>1.8).sum()),nparams=m.n_params(),n_events=len(evt),n_val_events=len(evv),rmse_nom=float(np.sqrt(np.nanmean((Zvn-pvn)**2))))
    if sd==0: np.savez("ex0.npz",Z=Zte,s=s_t,sad=adapt(s_t),flag=fl_ad,lab=lab[VA:],pred=pt,ev=np.array([(a,b) for a,b,_,_ in evt]))
    log(f"  done {time.time()-t0:.0f}s "+str({k:round(v['f1'],2) for k,v in res.items()}))
    json.dump(out,open("results_anomaly.json","w"))
