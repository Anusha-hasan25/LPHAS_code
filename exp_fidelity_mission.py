import numpy as np, json, zlib
from scipy.optimize import curve_fit
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"serif","font.size":9,"axes.grid":True,"grid.alpha":.3})
R=json.load(open("results_anomaly.json"))["seeds"]; out={}
k=np.arange(1,61).astype(float)
calm=np.mean([R[s]["calm"] for s in R],0); storm=np.mean([R[s]["storm"] for s in R],0)
# empirical fidelity F_k = exp(-NRMSE_k); test three growth laws for NRMSE_k
laws={"linear":(lambda k,l:l*k,[0.02]),"power":(lambda k,l,b:l*k**b,[0.1,0.7]),"saturating":(lambda k,a,t:a*(1-np.exp(-k/t)),[1.2,25])}
fits={}
for name,y in (("calm",calm),("storm",storm)):
    for lw,(f,p0) in laws.items():
        p,_=curve_fit(f,k,y,p0=p0,maxfev=20000); res=y-f(k,*p); rss=float((res**2).sum()); n=len(k); kk=len(p)
        aic=n*np.log(rss/n)+2*kk; r2=1-rss/((y-y.mean())**2).sum()
        fits[f"{name}_{lw}"]=dict(params=[float(x) for x in p],rss=rss,aic=float(aic),r2=float(r2))
        print(name,lw,np.round(p,4),"R2",round(r2,4),"AIC",round(aic,1))
out["fits"]=fits
# k* for Fmin=0.6 under each law  (threshold on NRMSE = -ln 0.6)
th=-np.log(0.6); ks={}
for name in ("calm","storm"):
    l=fits[f"{name}_linear"]["params"][0]; ks[f"{name}_linear"]=th/l
    l,b=fits[f"{name}_power"]["params"]; ks[f"{name}_power"]=(th/l)**(1/b)
    a,t=fits[f"{name}_saturating"]["params"]; ks[f"{name}_sat"]=(-t*np.log(1-th/a)) if th<a else float("inf")
    emp=np.argmax((calm if name=="calm" else storm)>=th)+1; ks[f"{name}_empirical"]=float(emp)
print("k*",ks); out["kstar"]=ks
fig,ax=plt.subplots(1,2,figsize=(7.0,2.6))
for a_,(name,y,c) in zip(ax,(("calm",calm,"#2E75B6"),("storm",storm,"#C00000"))):
    a_.plot(k,y,"o",ms=2.5,color=c,label="measured NRMSE$_k$")
    for lw,(f,_),ls in zip(laws,laws.values(),("--","-",":")):
        p=fits[f"{name}_{lw}"]["params"]; a_.plot(k,f(k,*p),ls,color="k",lw=1,label=f"{lw} (R$^2$={fits[f'{name}_{lw}']['r2']:.3f})")
    a_.axhline(th,color="gray",lw=.8); a_.set_title(name+" regime",fontsize=8); a_.set_xlabel("Rollout step $k$"); a_.legend(fontsize=6)
ax[0].set_ylabel("NRMSE of autonomous rollout"); plt.tight_layout(); plt.savefig("fig/f_rollout.png",dpi=300); plt.close()

# ---------------- mission co-simulation
rng=np.random.default_rng(99); AU=499.00478; td=15.0; to=0.5; eg=0.005; Cmax=240.0; D=200.0; Fmin_th=th
T=780; days=np.arange(T); thang=2*np.pi*days/T; d=np.sqrt(1+1.524**2-2*1.524*np.cos(thang)); Lmin=d*AU/60
H=np.full(T,0.05)
for c,w,a in [(150,12,0.55),(420,18,0.7),(610,10,0.45)]: H+=a*np.exp(-0.5*((days-c)/w)**2)
H=np.clip(H,0,1)
# fidelity growth law -> k* as function of hazard: interpolate between calm and storm power-law fits
lc,bc=fits["calm_power"]["params"]; ls_,bs=fits["storm_power"]["params"]
w=np.clip(H/0.6,0,1); lam=lc+(ls_-lc)*w; beta=bc+(bs-bc)*w; kstar=(th/lam)**(1/beta)
out["kstar_range_mission"]=[float(kstar.min()),float(kstar.max())]
def simulate(m_unval,runs=100,Ncls=40,rate=3.0,Cmed=300):
    acc={p:[] for p in ("Ground-only","Fixed envelope","Delay-indexed","Delay-indexed + rehearse","Rehearse + deadline override")}
    for _ in range(runs):
        f=rng.lognormal(0,1.0,Ncls); f/=f.sum()
        e=np.clip(rng.lognormal(np.log(0.03),0.8,Ncls),0.002,0.3); C=rng.lognormal(np.log(Cmed),1.3,Ncls)
        Ta=rng.lognormal(np.log(8),0.8,Ncls); allowed=C<=Cmax
        E3=((e-eg)*C<2*3+td)&allowed
        n=rng.poisson(rate*T); day=rng.integers(0,T,n); cls=rng.choice(Ncls,n,p=f); Tc=rng.lognormal(np.log(45),1.0,n)
        Tg=2*Lmin[day]+td; ei,Ci,Tai=e[cls],C[cls],Ta[cls]; al=allowed[cls]; ks=kstar[day]
        def outcome(deleg,unval):
            pw=np.where(deleg,ei*np.where(unval,m_unval,1.0),eg)
            resp=np.where(deleg,to,to+Tg); miss=(~deleg)&(Tc<Tg)
            cost=resp+pw*Ci+D*miss
            return dict(cost=cost.mean(),resp=resp.mean(),miss=miss.mean(),wrong=(np.where(deleg,pw,0).sum()/max(deleg.sum(),1)),deleg=deleg.mean())
        unval=Tai>ks
        g=np.zeros(n,bool); acc["Ground-only"].append(outcome(g,unval))
        acc["Fixed envelope"].append(outcome(E3[cls],unval))
        d2=al&(((ei-eg)*Ci<Tg)|(Tc<Tg)); acc["Delay-indexed"].append(outcome(d2,unval))
        d3=d2&~unval; acc["Delay-indexed + rehearse"].append(outcome(d3,unval))
        d4=d2&(~unval|(Tc<Tg)); acc["Rehearse + deadline override"].append(outcome(d4,unval))
    return {p:{k_:float(np.mean([r[k_] for r in v])) for k_ in v[0]} for p,v in acc.items()}
for m in (1,2,4,8):
    out[f"mission_m{m}"]=simulate(m); print("m",m,{p:(round(v["cost"],1),round(v["miss"],3),round(v["wrong"],3),round(v["deleg"],2)) for p,v in out[f"mission_m{m}"].items()})
# near vs far Mars (m=4)
# byte model
Xday=np.cumsum(rng.standard_normal((1440,8))*0.05,0)+np.sin(np.arange(1440)[:,None]/90*2*np.pi)
q=np.round(Xday/ (Xday.std()/200)).astype(np.int16); raw=q.nbytes
dl=np.diff(q,axis=0,prepend=q[:1]).astype(np.int16); zl=len(zlib.compress(dl.tobytes(),9))
pk=60; daily=3*pk+64
out["bytes"]=dict(raw_f32=1440*8*4,raw_int16=raw,zlib_delta=zl,narrative=daily)
print(out["bytes"]); print("kstar mission range",out["kstar_range_mission"])
json.dump(out,open("results_mission.json","w"))
# plot mission policies at m=4
pol=list(out["mission_m4"].keys()); fig,ax=plt.subplots(1,3,figsize=(7.2,2.5))
cols=["#7F7F7F","#BF8F00","#2E75B6","#548235","#C00000"]
for a_,key,lab in zip(ax,("cost","miss","wrong"),("Mean cost per event (min-eq)","Deadline-miss rate","Wrong-action rate (delegated)")):
    a_.bar(range(5),[out["mission_m4"][p][key] for p in pol],color=cols); a_.set_xticks(range(5)); a_.set_xticklabels(["Ground","Fixed","Delay-\nidx","+Reh.","+Reh.\n+DL"],fontsize=6.5); a_.set_title(lab,fontsize=8)
plt.tight_layout(); plt.savefig("fig/f_mission.png",dpi=300); plt.close()
