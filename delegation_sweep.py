import numpy as np, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"serif","font.size":9,"axes.grid":True,"grid.alpha":.3})
rng=np.random.default_rng(11)
N=40; td=15.0; eg=0.005
def run(Cmed,S=500):
    Lg=(3,12,22); out={}
    P={L:[] for L in Lg}; adapt={L:[] for L in Lg}; f3={L:[] for L in Lg}; f22={L:[] for L in Lg}
    curve=np.zeros((S,40)); Lc=np.linspace(0.5,22,40)
    for s in range(S):
        f=rng.lognormal(0,1.0,N); f/=f.sum()
        e=np.clip(rng.lognormal(np.log(0.03),0.8,N),0.002,0.3)
        C=rng.lognormal(np.log(Cmed),1.3,N)
        dm=lambda L:(e-eg)*C<2*L+td
        cost=lambda L,m:float((f*np.where(m,e*C,2*L+td+eg*C)).sum())
        m3,m22=dm(3),dm(22)
        for L in Lg:
            ma=dm(L); P[L].append(f[ma].sum()); adapt[L].append(cost(L,ma)); f3[L].append(cost(L,m3)); f22[L].append(cost(L,m22))
        for j,L in enumerate(Lc): curve[s,j]=f[dm(L)].sum()
    for L in Lg:
        out[L]=dict(pE=round(float(np.mean(P[L])),2),adaptive=round(float(np.mean(adapt[L])),1),fixed3=round(float(np.mean(f3[L])),1),fixed22=round(float(np.mean(f22[L])),1))
    return out,Lc,curve
allres={}; curves={}
for Cm in (30,100,300,1000):
    o,Lc,cv=run(Cm); allres[Cm]=o; curves[Cm]=cv
    print(Cm,o)
json.dump({str(k):{str(l):v for l,v in d.items()} for k,d in allres.items()},open("sweep.json","w"))
fig,ax=plt.subplots(figsize=(3.4,2.5))
cols={30:"#548235",100:"#2E75B6",300:"#BF8F00",1000:"#C00000"}
for Cm,cv in curves.items():
    ax.plot(Lc,cv.mean(0),color=cols[Cm],label=f"median $C$ = {Cm} min-eq")
ax.set_xlabel("One-way delay (min)"); ax.set_ylabel("Delegated share $p_E(L)$"); ax.set_ylim(0,1.02); ax.legend(fontsize=6.5,loc="center right")
plt.tight_layout(); plt.savefig("fig/f5_deleg.png",dpi=300); plt.close()
