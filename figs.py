import numpy as np, json
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({"font.family":"serif","font.size":9,"axes.grid":True,"grid.alpha":.3})
R=json.load(open("results_anomaly.json"))["seeds"]; seeds=sorted(R)
def box(ax,x,y,w,h,t,fc="#D9E2F3",ec="#2F5597",fs=7.5,bold=False):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.02,rounding_size=0.08",fc=fc,ec=ec,lw=1))
    ax.text(x+w/2,y+h/2,t,ha="center",va="center",fontsize=fs,fontweight="bold" if bold else "normal")
def arr(ax,x1,y1,x2,y2,c="k",ls="-",t=None,tx=0,ty=0):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=9,lw=1,color=c,ls=ls))
    if t: ax.text((x1+x2)/2+tx,(y1+y2)/2+ty,t,fontsize=6.5,ha="center",color=c)
# ---- architecture
fig,ax=plt.subplots(figsize=(7.2,4.3)); ax.set_xlim(0,12); ax.set_ylim(0,7.2); ax.axis("off"); ax.grid(False)
ax.add_patch(FancyBboxPatch((0.1,0.2),8.0,6.8,boxstyle="round,pad=0.02,rounding_size=0.1",fc="#F7F7F7",ec="#555",lw=1.2,ls="--"))
ax.text(4.1,6.75,"Spacecraft / habitat (onboard, 10 W, 64 MB class)",ha="center",fontsize=8,fontweight="bold")
box(ax,0.4,5.0,1.9,1.1,"Sensors\n& actuators",fc="#E2EFDA",ec="#548235")
box(ax,2.8,5.0,2.0,1.1,"Tier 1\nData Gate\n(validity, impute)",fc="#D9E2F3")
box(ax,5.3,5.0,2.5,1.1,"Hazard index $H(t)$\nweighted indicators",fc="#FFF2CC",ec="#BF8F00")
box(ax,0.4,3.0,2.4,1.3,"Tier 2a\nAnomaly engine\nLSTM + NDT",fc="#D9E2F3")
box(ax,3.1,3.0,2.4,1.3,"Tier 2b\nShadow twin\nrollout, $k^{*}$",fc="#D9E2F3")
box(ax,5.8,3.0,2.1,1.3,"Tier 2c\nCausal module\n(PC, event-triggered)",fc="#D9E2F3")
box(ax,0.4,1.0,3.3,1.3,"Policy: $A_c$, AL0–AL4,\nhysteresis, hazard-gated AL4\ndelegation rule $E(L)$, ledger $K$",fc="#FCE4D6",ec="#C55A11",bold=True)
box(ax,4.1,1.0,2.0,1.3,"Rehearse-then-\ncommit gate",fc="#FCE4D6",ec="#C55A11")
box(ax,6.4,1.0,1.6,1.3,"Tier 3\nNarrative\npacket",fc="#E2EFDA",ec="#548235")
box(ax,9.0,3.0,2.8,1.5,"Ground control\n(operators, $t_d$)",fc="#EDEDED",ec="#555",bold=True)
box(ax,9.0,5.3,2.8,1.1,"Ledger review\n(outcome judgement)",fc="#EDEDED",ec="#555")
arr(ax,2.3,5.55,2.8,5.55); arr(ax,4.8,5.55,5.3,5.55)
arr(ax,3.8,5.0,1.6,4.3); arr(ax,3.8,5.0,4.3,4.3); arr(ax,6.5,5.0,6.8,4.3)
arr(ax,1.6,3.0,1.9,2.3); arr(ax,4.3,3.0,4.9,2.3); arr(ax,6.8,3.0,7.2,2.3)
arr(ax,6.5,5.0,3.0,2.3,c="#BF8F00",ls="--")
arr(ax,3.7,1.65,4.1,1.65); arr(ax,6.1,1.65,6.4,1.65)
arr(ax,0.9,1.0,0.9,0.55,c="#548235"); ax.text(1.0,0.35,"actions within $E(L)$",fontsize=6.5,color="#548235")
arr(ax,8.0,1.65,9.4,3.0,c="#548235",t="narrative downlink (delay $L$)",tx=1.2,ty=-0.7)
arr(ax,9.3,3.0,8.0,1.9,c="#555",ls="--",t="uplink: veto, $E$, ledger",tx=-0.3,ty=0.55)
arr(ax,10.4,4.5,10.4,5.3,c="#555")
plt.tight_layout(); plt.savefig("fig/d_arch.png",dpi=300); plt.close()
# ---- state machine
fig,ax=plt.subplots(figsize=(7.0,2.6)); ax.set_xlim(0,12); ax.set_ylim(0,4.3); ax.axis("off"); ax.grid(False)
xs=[0.2,2.6,5.0,7.4,9.8]; names=["AL0\nground only","AL1\nonboard proposes","AL2\nact in $E$, report","AL3\nact in $E$, ground veto","AL4\nemergency authority"]
cs=["#EDEDED","#D9E2F3","#BDD7EE","#9DC3E6","#F4B183"]
for x,n,c in zip(xs,names,cs): box(ax,x,1.6,1.55,1.3,n,fc=c,ec="#2F5597",fs=6.3)
for i in range(4):
    arr(ax,xs[i]+1.55,2.5,xs[i+1],2.5); ax.text(xs[i]+1.55+0.45,2.75,f"$\\geq${0.2*(i+1)+0.03:.2f}",fontsize=6,ha="center"); arr(ax,xs[i+1],1.95,xs[i]+1.55,1.95,c="#7F7F7F"); ax.text(xs[i]+1.55+0.45,1.45,f"$<${0.2*(i+1)-0.03:.2f}",fontsize=6,ha="center",color="#7F7F7F")
ax.text(10.6,3.2,"requires $H\\geq 0.5$ (else stay AL3)",ha="center",fontsize=7,color="#C55A11"); ax.text(6,3.9,"Thresholds on $A_c$ (up / down with hysteresis 0.03)",ha="center",fontsize=7.5)
ax.text(6,0.7,"Hysteresis margin 0.03 on every threshold; fidelity or Data-Gate failure → safe mode + priority packet (not AL0)",ha="center",fontsize=7.5)
plt.tight_layout(); plt.savefig("fig/d_states.png",dpi=300); plt.close()
# ---- space-time diagram
fig,ax=plt.subplots(1,2,figsize=(7.2,3.2),sharey=True)
Lm=12; td=15; ev=0
for a_,title in zip(ax,("Ground-in-the-loop","LPHAS (in-envelope event)")):
    a_.set_xlim(0,3); a_.set_ylim(0,2*Lm+td+8); a_.set_xticks([0.5,2.5]); a_.set_xticklabels(["Spacecraft","Earth"]); a_.set_title(title,fontsize=8); a_.invert_yaxis()
    a_.axvline(0.5,color="k",lw=1.5); a_.axvline(2.5,color="k",lw=1.5)
ax[0].set_ylabel("Time (min)")
a=ax[0]; a.annotate("",xy=(2.5,Lm),xytext=(0.5,0),arrowprops=dict(arrowstyle="-|>",color="#C00000")); a.text(1.5,Lm/2-1,"anomaly report",ha="center",fontsize=7,color="#C00000",rotation=-15)
a.plot([2.5,2.5],[Lm,Lm+td],color="#BF8F00",lw=5); a.text(2.55,Lm+td/2,"$t_d$",fontsize=7,va="center")
a.annotate("",xy=(0.5,2*Lm+td),xytext=(2.5,Lm+td),arrowprops=dict(arrowstyle="-|>",color="#2E75B6")); a.text(1.5,Lm+td+Lm/2+1,"command",ha="center",fontsize=7,color="#2E75B6",rotation=15)
a.plot(0.5,2*Lm+td,"o",color="g"); a.text(0.6,2*Lm+td+3,f"response at {2*Lm+td} min",fontsize=7)
b=ax[1]; b.plot([0.5,0.5],[0,0.5],color="g",lw=5); b.text(0.6,-1.2,"onboard response at 0.5 min",fontsize=7)
b.annotate("",xy=(2.5,Lm),xytext=(0.5,0.5),arrowprops=dict(arrowstyle="-|>",color="#548235")); b.text(1.5,Lm/2,"narrative packet\n(+ counterfactual)",ha="center",fontsize=7,color="#548235",rotation=-15)
b.plot([2.5,2.5],[Lm,Lm+4],color="#BF8F00",lw=5); b.text(2.55,Lm+2,"audit",fontsize=7,va="center")
b.annotate("",xy=(0.5,2*Lm+4),xytext=(2.5,Lm+4),arrowprops=dict(arrowstyle="-|>",color="#7F7F7F",ls="--")); b.text(1.5,Lm+4+Lm/2,"veto / revised $E$",ha="center",fontsize=7,color="#7F7F7F",rotation=15)
for a_ in ax: a_.grid(False)
plt.tight_layout(); plt.savefig("fig/d_spacetime.png",dpi=300); plt.close()
# ---- packet layout picture
fig,ax=plt.subplots(figsize=(7.2,2.6)); ax.set_xlim(0,100); ax.set_ylim(0,10); ax.axis("off"); ax.grid(False)
parts=[("Header 16 B\n$t$, $A_c$, AL,\n$H$, $F$",0,21,"#D9E2F3"),("Event record 24 B\nclass, cause,\naction, conf.",21,24,"#E2EFDA"),("Counterfactual 16 B\nrejected alt.,\n$\\Delta$ outcome",45,23,"#FFF2CC"),("Template 4 B\nID + slots",68,17,"#FCE4D6"),("CRC 4 B\nseq.",85,15,"#EDEDED")]
for t,x,w,c in parts: box(ax,x+0.3,4.2,w-0.8,4.6,t,fc=c,fs=6.3)
ax.text(50,2.7,"Example rendered packet (about 64 B on the wire):",ha="center",fontsize=7,fontweight="bold")
ax.text(50,1.3,"“t = day 412: dust-opacity rise. Cause (p = 0.82): panel soiling. Acted: reduced load 20%.\nRejected: full-load hold (predicted shortfall 3.1 Wh).”",ha="center",fontsize=6.8)
plt.tight_layout(); plt.savefig("fig/d_packet.png",dpi=300); plt.close()
# ---- Ac heatmap
Lr=np.linspace(0,120,200); Hh=np.linspace(0,1,100); TAU=1800
fig,ax=plt.subplots(1,3,figsize=(7.2,2.5),sharey=True)
for a_,K in zip(ax,(0.8,1.2,1.8)):
    G=np.minimum(1,(1-np.exp(-Lr[None,:]*60/TAU))*(1+Hh[:,None])/K)
    al=np.digitize(G,[.2,.4,.6,.8]); al=np.where((al==4)&(Hh[:,None]<0.5),3,al)
    im=a_.pcolormesh(Lr,Hh,al,cmap="Blues",shading="auto",vmin=0,vmax=4.6); a_.set_title(f"$K$={K}",fontsize=8); a_.set_xlabel("Round-trip delay (min)"); a_.grid(False)
    for x_,l_ in ((8.7,"Mars\nnear"),(42,"Mars\nfar"),(100,"Jupiter")): a_.axvline(x_,color="k",lw=.6,ls=":")
ax[0].set_ylabel("Hazard $H$"); cb=fig.colorbar(im,ax=ax,ticks=[0,1,2,3,4],fraction=0.03); cb.set_label("AL",fontsize=8)
plt.savefig("fig/f_heat.png",dpi=300,bbox_inches="tight"); plt.close()
# ---- training curves
tr=np.array([R[s]["hist"]["train"] for s in seeds]); va=np.array([R[s]["hist"]["val"] for s in seeds]); ep=np.arange(1,tr.shape[1]+1)
fig,ax=plt.subplots(figsize=(3.6,2.5))
ax.plot(ep,tr.mean(0),color="#2E75B6",label="train MSE"); ax.fill_between(ep,tr.min(0),tr.max(0),color="#2E75B6",alpha=.2)
ax.plot(ep,va.mean(0),color="#C00000",label="validation MSE"); ax.fill_between(ep,va.min(0),va.max(0),color="#C00000",alpha=.2)
ax.set_yscale("log"); ax.set_xlabel("Epoch"); ax.set_ylabel("MSE (z-scored)"); ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig("fig/f_train.png",dpi=300); plt.close()
# ---- detection example
d=np.load("ex0.npz"); Z,pr,s,sad,fl,lab,evs=d["Z"],d["pred"],d["s"],d["sad"],d["flag"],d["lab"],d["ev"]
hit=[i for i,(a_,b_) in enumerate(evs) if fl[a_:b_].any()]; i0=hit[1] if len(hit)>1 else (hit[0] if hit else 3); a0=max(0,int(evs[i0][0])-400); sl=slice(a0,a0+1600); err=np.abs(Z-pr); ch=int(np.nanargmax(err[int(evs[i0][0]):int(evs[i0][1])].max(0)))
fig,ax=plt.subplots(3,1,figsize=(7.0,4.0),sharex=True); t=np.arange(len(Z))[sl]
ax[0].plot(t,Z[sl,ch],color="k",lw=.8,label="observed"); ax[0].plot(t,pr[sl,ch],color="#2E75B6",lw=.8,label="LSTM forecast"); ax[0].set_ylabel(f"ch {ch} (z)"); ax[0].legend(fontsize=6,ncol=2,loc="upper right")
ax[1].plot(t,s[sl],color="#7F7F7F",lw=.8,label="score $s_t$"); ax[1].plot(t,sad[sl],color="#C55A11",lw=.8,label="adaptive score"); ax[1].legend(fontsize=6,loc="upper right"); ax[1].set_ylabel("score")
ax[2].fill_between(t,0,lab[sl],color="#C00000",alpha=.4,step="mid",label="true anomaly"); ax[2].fill_between(t,0,fl[sl]*0.6,color="#2E75B6",alpha=.6,step="mid",label="detected"); ax[2].legend(fontsize=6,loc="upper right"); ax[2].set_yticks([]); ax[2].set_xlabel("Test step")
plt.tight_layout(); plt.savefig("fig/f_detect.png",dpi=300); plt.close()
# ---- comparison bars
methods=list(R["0"]["res"].keys()); lab_=["LSTM\n+NDT","LSTM\n+fixed","LSTM+adapt\n+NDT","Persist.\n+NDT","Isolation\nForest"]
fig,ax=plt.subplots(1,2,figsize=(7.2,2.6)); w=0.27
for j,(k,c) in enumerate((("precision","#2E75B6"),("recall","#BF8F00"),("f1","#548235"))):
    m=[np.mean([R[s]["res"][me][k] for s in seeds]) for me in methods]; sd=[np.std([R[s]["res"][me][k] for s in seeds]) for me in methods]
    ax[0].bar(np.arange(5)+(j-1)*w,m,w,yerr=sd,color=c,label=k,capsize=1.5,error_kw=dict(lw=.7))
ax[0].set_xticks(range(5)); ax[0].set_xticklabels(lab_,fontsize=6); ax[0].legend(fontsize=6); ax[0].set_title("Event-wise metrics, mean ± SD (5 seeds)",fontsize=8)
types=["spike","level","drift","freq","stuck","noise"]; pt=[np.mean([R[s]["ptype"].get(t_,0) for s in seeds]) for t_ in types]
ax[1].bar(types,pt,color="#2E75B6"); ax[1].set_ylim(0,1); ax[1].set_title("Recall by anomaly type (LSTM+adapt+NDT)",fontsize=8); ax[1].tick_params(axis="x",labelsize=7)
plt.tight_layout(); plt.savefig("fig/f_compare.png",dpi=300); plt.close()
# ---- quantisation
bits=["fp32","8","6","4","3"]; f1=[np.mean([R[s]["quant"][b]["f1"] for s in seeds]) for b in bits]; rm=[np.mean([R[s]["quant"][b]["rmse"] for s in seeds]) for b in bits]; by=[R["0"]["quant"][b]["bytes"]/1024 for b in bits]
fig,ax=plt.subplots(figsize=(3.6,2.5)); x=np.arange(5)
ax.bar(x-0.2,f1,0.4,color="#2E75B6",label="event F1"); ax.set_ylabel("Event F1"); ax.set_xticks(x); ax.set_xticklabels(["FP32","INT8","INT6","INT4","INT3"])
a2=ax.twinx(); a2.plot(x,by,"o-",color="#C00000",label="size (KB)"); a2.set_ylabel("Weights (KB)"); a2.grid(False)
h1,l1=ax.get_legend_handles_labels(); h2,l2=a2.get_legend_handles_labels(); ax.legend(h1+h2,l1+l2,fontsize=6.5,loc="upper right")
plt.tight_layout(); plt.savefig("fig/f_quant.png",dpi=300); plt.close()
print("figs ok")
