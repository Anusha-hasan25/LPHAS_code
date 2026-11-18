import numpy as np, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"serif","font.size":9,"axes.grid":True,"grid.alpha":.3})
AU_S = 499.00478  # light-time per AU (s)

# ---- 1. light-time (circular coplanar approx. for planets; Moon fixed)
def dist_range(a):  # Earth r=1 AU circular
    return abs(a-1.0), a+1.0
bodies = {"Moon":(0.00257,0.00257),"Mars":dist_range(1.524),"Jupiter":dist_range(5.203),"Saturn":dist_range(9.537)}
ow = {k:(v[0]*AU_S, v[1]*AU_S) for k,v in bodies.items()}
for k,v in ow.items(): print(k,"one-way s",[round(x,1) for x in v],"min",[round(x/60,1) for x in v])
json.dump({k:[v[0],v[1]] for k,v in ow.items()},open("lt.json","w"))

fig,ax=plt.subplots(figsize=(3.4,2.4))
names=list(ow); x=np.arange(len(names))
lo=[ow[n][0]/60 for n in names]; hi=[ow[n][1]/60 for n in names]
ax.bar(x,[h-l for l,h in zip(lo,hi)],bottom=lo,color="#4472C4",width=.55)
ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylabel("One-way delay (min, log)")
for i,(l,h) in enumerate(zip(lo,hi)): ax.text(i,h*1.12,f"{l:.1f}–{h:.1f}" if l>0.5 else f"{h*60:.1f} s",ha="center",fontsize=7)
ax.set_ylim(0.01,300); plt.tight_layout(); plt.savefig("fig/f1_delay.png",dpi=300); plt.close()

# ---- 2. Autonomy coefficient (bounded, dimensionless)
TAU=1800.0; GAMMA=1.0
def Lhat(Lrt): return 1-np.exp(-Lrt/TAU)
def Ac(Lrt,H,K): return np.minimum(1.0, Lhat(Lrt)*(1+GAMMA*H)/K)
def AL(a): return np.digitize(a,[0.2,0.4,0.6,0.8])

# Mars synodic cycle, circular coplanar
T=780; t=np.arange(0,T+1)
th=2*np.pi*t/T
d=np.sqrt(1+1.524**2-2*1.524*np.cos(th))      # AU
Lrt=2*d*AU_S
print("Mars RT min range",Lrt.min()/60,Lrt.max()/60)
# synthetic hazard: baseline 0.05 plus 3 storm episodes (assumption, documented)
rng=np.random.default_rng(7)
H=np.full(T+1,0.05)
for c,w,a in [(150,12,0.55),(420,18,0.7),(610,10,0.45)]:
    H+=a*np.exp(-0.5*((t-c)/w)**2)
H=np.clip(H+rng.normal(0,0.01,T+1),0,1)
fig,axs=plt.subplots(2,1,figsize=(3.4,3.6),sharex=True,gridspec_kw={"height_ratios":[1,2]})
axs[0].plot(t,Lrt/60,color="k",lw=1,label="Round-trip delay (min)")
ax2=axs[0].twinx(); ax2.plot(t,H,color="#C00000",lw=.8,label="H(t)"); ax2.set_ylim(0,1); ax2.grid(False)
axs[0].set_ylabel("RTT (min)"); ax2.set_ylabel("H(t)",color="#C00000")
for K,c in [(0.8,"#2E75B6"),(1.2,"#548235"),(1.8,"#BF8F00")]:
    axs[1].plot(t,Ac(Lrt,H,K),color=c,lw=1,label=f"$K_{{trust}}$={K}")
for y in [.2,.4,.6,.8]: axs[1].axhline(y,color="gray",ls=":",lw=.7)
axs[1].set_ylabel("$A_c$"); axs[1].set_xlabel("Day of Earth–Mars synodic cycle"); axs[1].set_ylim(0,1.02)
axs[1].legend(fontsize=7,loc="lower right",ncol=3)
plt.tight_layout(); plt.savefig("fig/f2_ac.png",dpi=300); plt.close()
# fraction of time at each AL
res={}
for K in (0.8,1.2,1.8):
    a=Ac(Lrt,H,K); al=AL(a); res[K]=[float((al==i).mean()*100) for i in range(5)]
    print("K",K,[round(x,1) for x in res[K]])
json.dump(res,open("al_frac.json","w"))

# per-destination AL table (nominal H=0.05, storm H=0.6), K=1.2
tab=[]
for n,(lo_,hi_) in ow.items():
    row=[n]
    for Hh in (0.05,0.6):
        a_lo=float(Ac(2*lo_,Hh,1.2)); a_hi=float(Ac(2*hi_,Hh,1.2))
        row.append((a_lo,a_hi,int(AL(a_lo)),int(AL(a_hi))))
    tab.append(row); print(row)
json.dump(tab,open("dest_tab.json","w"))

# threshold sensitivity: perturb thresholds +-0.05 -> fraction of time AL changes (Mars, K=1.2)
a=Ac(Lrt,H,1.2); base=AL(a)
ch=[]
for s in (-0.05,0.05):
    thr=np.array([.2,.4,.6,.8])+s; al2=np.digitize(a,thr); ch.append(float((al2!=base).mean()*100))
print("threshold shift +-0.05 -> % time AL differs:",ch)
json.dump(ch,open("sens.json","w"))

# ---- 3. fidelity decay
k=np.arange(0,101)
fig,ax=plt.subplots(figsize=(3.4,2.4))
ks={}
for lam,lab,c in [(0.01,"stable cruise","#548235"),(0.05,"nominal","#2E75B6"),(0.12,"dust storm","#C00000")]:
    ax.plot(k,np.exp(-lam*k),color=c,label=f"{lab} ($\\lambda$={lam})")
    ks[lam]=-np.log(0.6)/lam; print("lam",lam,"k*",ks[lam])
ax.axhline(0.6,color="k",ls="--",lw=.8); ax.text(70,0.62,"$F_{min}$=0.6",fontsize=7)
ax.set_xlabel("Prediction horizon k (timesteps)"); ax.set_ylabel("Shadow fidelity $F_k$"); ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig("fig/f3_fid.png",dpi=300); plt.close()
json.dump({str(k_):v for k_,v in ks.items()},open("kstar.json","w"))

# ---- 4. analytic MTAR
Ld=np.linspace(0.5,22,200)*60  # one-way s
t_o=30.0; t_d=15*60.0   # onboard handling 30 s; ground decision 15 min (assumed)
fig,ax=plt.subplots(figsize=(3.4,2.4))
base=t_o+2*Ld+t_d
ax.plot(Ld/60,base/60,"k",label="Ground-in-the-loop")
for p,c in [(0.5,"#BF8F00"),(0.7,"#2E75B6"),(0.9,"#548235")]:
    m=p*t_o+(1-p)*(t_o+2*Ld+t_d)
    ax.plot(Ld/60,m/60,color=c,label=f"LPHAS, $p_E$={p}")
ax.set_xlabel("One-way delay (min)"); ax.set_ylabel("Expected MTAR (min)"); ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig("fig/f4_mtar.png",dpi=300); plt.close()
for L_ in (3,12,22):
    Lx=L_*60; b=t_o+2*Lx+t_d
    print("L",L_,"base min",round(b/60,1),[ (p,round((p*t_o+(1-p)*b)/60,1), round(100*(1-(p*t_o+(1-p)*b)/b),1)) for p in (0.5,0.7,0.9)])

# ---- 5. NCR sensitivity
for dl in (0.5,0.7,0.85,0.9,0.95): print("delta",dl,"NCR",round(1/(1-dl),2))

# ---- 6. power analysis
from statsmodels.stats.power import TTestPower
tp=TTestPower()
for d_ in (0.3,0.5,0.8):
    print("d",d_,"N",np.ceil(tp.solve_power(effect_size=d_,alpha=0.05,power=0.8)))
print("N=18 detectable d at 80% power:",tp.solve_power(nobs=18,alpha=0.05,power=0.8))
# power budget
print(0.9+2.1+4.5+0.8, 0.9+2.1+4.5+0.8+1.2)
