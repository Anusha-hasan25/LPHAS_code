import numpy as np, json
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"serif","font.size":9,"axes.grid":True,"grid.alpha":.3})
rng=np.random.default_rng(5); R=2000; N=400; r0=0.9
def Tmap(r): return np.maximum(0,(r-r0)/(1-r0))
def run(rtrue,gamma=1.0):
    """returns K trajectories for: LCB(cumul or discounted), posterior mean, point estimate"""
    succ=(rng.random((R,N))<rtrue[None,:]).astype(float)
    s=np.zeros(R); n=np.zeros(R); K={m:np.zeros((R,N+1)) for m in ("LCB","mean","point")}
    for m in K: K[m][:,0]=1.8
    for j in range(N):
        s=gamma*s+succ[:,j]; n=gamma*n+1
        a,b=1+s,1+n-s
        K["LCB"][:,j+1]=1.8-Tmap(stats.beta.ppf(0.05,a,b))
        K["mean"][:,j+1]=1.8-Tmap(a/(a+b)); K["point"][:,j+1]=1.8-Tmap(s/n)
    return K
out={}
# S1 good agent r=.97 ; S2 marginal r=.90 ; S3 degradation at n=100 (0.97 -> 0.85)
r1=np.full(N,0.97); r2=np.full(N,0.90); r3=np.where(np.arange(N)<200,0.97,0.85)
K1=run(r1); K2=run(r2); K3=run(r3); K3d=run(r3,gamma=0.98)
thr=1.4
for m in ("LCB","mean","point"):
    first=np.array([np.argmax(k<=thr) if (k<=thr).any() else np.nan for k in K1[m]])
    out[f"S1_time_to_trust_{m}"]=float(np.nanmedian(first)); out[f"S1_reach_{m}"]=float(np.mean(~np.isnan(first)))
    out[f"S2_false_trust_{m}"]=float(np.mean((K2[m]<=thr).any(1)))
    out[f"S2_false_trust_final_{m}"]=float(np.mean(K2[m][:,-1]<=thr))
GAM=[1.0,0.995,0.99,0.98]
Ks={g:run(r3,gamma=g)["LCB"] for g in GAM}; K1s={g:run(r1,gamma=g)["LCB"] for g in GAM}
for g in GAM:
    Kk=Ks[g]; after=Kk[:,201:]
    dly=np.array([np.argmax(a_>=1.6) if (a_>=1.6).any() else np.nan for a_ in after])
    out[f"g{g}_K200"]=float(np.median(K1s[g][:,200])); out[f"g{g}_earned_frac"]=float(np.mean(K1s[g][:,200]<=1.4))
    out[f"g{g}_detect_median"]=float(np.nanmedian(dly)); out[f"g{g}_detect_frac"]=float(np.mean(~np.isnan(dly)))
print(out); json.dump(out,open("results_ledger.json","w"))
x=np.arange(N+1); fig,ax=plt.subplots(1,3,figsize=(7.2,2.5),sharey=True)
def band(a,K,c,l):
    a.plot(x,np.median(K,0),color=c,label=l); a.fill_between(x,np.percentile(K,10,0),np.percentile(K,90,0),color=c,alpha=.15)
band(ax[0],K1["LCB"],"#2E75B6","Ledger (5% LCB)"); band(ax[0],K1["mean"],"#BF8F00","Posterior mean"); band(ax[0],K1["point"],"#C00000","Point estimate"); ax[0].set_title("S1: reliable agent ($r$=0.97)",fontsize=8)
band(ax[1],K2["LCB"],"#2E75B6","Ledger (5% LCB)"); band(ax[1],K2["mean"],"#BF8F00","Posterior mean"); band(ax[1],K2["point"],"#C00000","Point estimate"); ax[1].set_title("S2: marginal agent ($r$=0.90)",fontsize=8)
band(ax[2],Ks[1.0],"#2E75B6","Cumulative ($\\gamma$=1)"); band(ax[2],Ks[0.99],"#548235","Discounted ($\\gamma$=0.99)"); ax[2].axvline(200,color="k",ls="--",lw=.8); ax[2].set_title("S3: degradation at $n$=200",fontsize=8)
for a in ax: a.set_xlabel("Delegated actions $n$")
ax[0].set_ylabel("$K$ (median, 10–90%)"); ax[0].legend(fontsize=6); ax[2].legend(fontsize=6); ax[1].legend(fontsize=6)
plt.tight_layout(); plt.savefig("fig/f_ledger.png",dpi=300); plt.close()
