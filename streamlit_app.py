
from __future__ import annotations

import calendar as cal
from datetime import date
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import streamlit as st
import yfinance as yf
from market_analyzer import (

    september_decline_probability,

    calculate_cycles,

    market_risk_score,

    risk_label,

)
STOCKS = ["AAPL","MSFT","NVDA","META","IBM","AMZN","GOOGL","AMD","MU"]
MARKET = ["SPY","QQQ"]
VOL = ["^VIX","^VIX3M"]
SECTORS = ["XLK","XLY","XLC","XLF","XLI","XLP","XLV","XLE","XLU","XLRE","XLB"]

st.set_page_config(page_title="September Market Risk", layout="wide")

st.markdown("""
<style>
.block-container{padding-top:1rem;max-width:1400px}
div[data-testid="stMetric"]{border:1px solid rgba(128,128,128,.25);border-radius:12px;padding:10px}
@media(max-width:700px){
.block-container{padding-left:.6rem;padding-right:.6rem}
h1{font-size:1.55rem!important}
.riskcal{gap:3px!important}
.rc{min-height:78px!important;padding:4px!important;border-radius:7px!important}
.rd{font-size:14px!important}.rs{font-size:9px!important}
}
</style>
""", unsafe_allow_html=True)

st.title("September Seasonality + Market Risk")
st.caption("Mobile-friendly research dashboard using stock seasonality, SPY/QQQ cycles, VIX structure and breadth.")

def dl(symbol, start="2000-01-01"):
    try:
        d = yf.download(symbol, start=start, auto_adjust=True, progress=False, actions=False)
        if d.empty:
            return pd.DataFrame()
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        d.index = pd.to_datetime(d.index).tz_localize(None)
        return d.sort_index()
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False)
def get_data(symbols):
    return {s: dl(s, "1980-01-01" if s in STOCKS else "2000-01-01") for s in symbols}

def september_stats(df, years, sep_day):
    if df.empty or "Close" not in df:
        return np.nan, np.nan, 0
    x = df.copy()
    x["ret"] = x["Close"].pct_change()
    s = x[x.index.month == 9].copy()
    if s.empty:
        return np.nan, np.nan, 0
    s["year"] = s.index.year
    s["td"] = s.groupby("year").cumcount() + 1
    if years:
        last = int(s["year"].max())
        s = s[s["year"] >= last-years+1]
    q = s[s["td"] == sep_day]["ret"].dropna()
    if len(q) == 0:
        return np.nan, np.nan, 0
    return float((q < 0).mean()*100), float(q.mean()*100), len(q)

def blended_seasonality(df, sep_day, weights):
    vals = []
    rets = []
    for label, years in [("5Y",5),("10Y",10),("20Y",20),("FULL",None)]:
        p, r, n = september_stats(df, years, sep_day)
        if pd.notna(p):
            reliability = min(n/5, 1)
            w = weights[label]*reliability
            vals.append((p,w))
            rets.append((r,w))
    if not vals:
        return np.nan, np.nan
    den = sum(w for _,w in vals) or 1
    denr = sum(w for _,w in rets) or 1
    return sum(v*w for v,w in vals)/den, sum(v*w for v,w in rets)/denr

def sigmoid(x, scale=1):
    return 1/(1+math.exp(-x/scale))

def cycle_risk(data):
    parts=[]
    for s in MARKET:
        d=data.get(s,pd.DataFrame())
        if d.empty: continue
        c=d["Close"].dropna()
        vals=[]
        for n,w in [(20,.25),(40,.30),(100,.25),(300,.20)]:
            if len(c)<=n: continue
            ret=(float(c.iloc[-1])/float(c.iloc[-n-1])-1)*100
            ma=float(c.rolling(n).mean().iloc[-1])
            dma=(float(c.iloc[-1])/ma-1)*100
            ext=100*sigmoid(.55*ret+.45*dma,8)
            brk=100*sigmoid(-dma,2.2)
            comp=(.45*ext+.55*brk) if n in (20,40) else (.70*ext+.30*brk)
            vals.append((comp,w))
        if vals:
            parts.append(sum(v*w for v,w in vals)/sum(w for _,w in vals))
    return float(np.mean(parts)) if parts else 50.0

def vol_snapshot(data):
    v=data.get("^VIX",pd.DataFrame())
    v3=data.get("^VIX3M",pd.DataFrame())
    if v.empty:
        return np.nan,np.nan,"Unavailable",50.0
    vix=float(v["Close"].dropna().iloc[-1])
    ratio=np.nan
    status="Unavailable"
    if not v3.empty:
        v3m=float(v3["Close"].dropna().iloc[-1])
        if v3m:
            ratio=vix/v3m
            status="Backwardation" if ratio>1 else "Contango"
    level=100*sigmoid(vix-20,5)
    struct=50 if pd.isna(ratio) else 100*sigmoid(ratio-1,.08)
    return vix,ratio,status,.65*level+.35*struct

def breadth_risk(data, selected):
    universe=list(dict.fromkeys(SECTORS+selected))
    scores=[]
    pcts={}
    for n,w in [(20,.40),(50,.35),(200,.25)]:
        flags=[]
        for s in universe:
            d=data.get(s,pd.DataFrame())
            if d.empty: continue
            c=d["Close"].dropna()
            if len(c)>=n:
                ma=c.rolling(n).mean().iloc[-1]
                if pd.notna(ma): flags.append(float(c.iloc[-1])>float(ma))
        pct=100*np.mean(flags) if flags else np.nan
        pcts[n]=pct
        if pd.notna(pct): scores.append((100-pct,w))
    risk=sum(v*w for v,w in scores)/sum(w for _,w in scores) if scores else 50
    return float(risk),pcts

def pc_risk(total_pc, equity_pc):
    vals=[]
    for x,lo,hi,scale in [(total_pc,.70,1.15,.12),(equity_pc,.55,.90,.10)]:
        comp=100*sigmoid(lo-x,scale)
        stress=100*sigmoid(x-hi,scale)
        vals.append(max(comp,stress))
    return float(np.mean(vals))

def risk_label(x):
    if x>=75:return "VERY HIGH"
    if x>=62:return "HIGH"
    if x>=50:return "MODERATE"
    if x>=38:return "LOW"
    return "VERY LOW"

def color(x):
    if x>=75:return "#ef4444"
    if x>=62:return "#f97316"
    if x>=50:return "#eab308"
    if x>=38:return "#84cc16"
    return "#22c55e"

with st.sidebar:
    selected=st.multiselect("Stocks", STOCKS, STOCKS)
    year=st.number_input("September year", 2000, 2100, 2026)
    st.markdown("#### Seasonality blend")
    w5=st.slider("5-year",0,100,35,5)
    w10=st.slider("10-year",0,100,30,5)
    w20=st.slider("20-year",0,100,25,5)
    wf=st.slider("Full history",0,100,10,5)
    st.markdown("#### Final score")
    sshare=st.slider("Seasonality share",0,100,55,5)
    st.markdown("#### Put/Call")
    usepc=st.checkbox("Use manual put/call",False)
    tpc=st.number_input("Total put/call",.10,3.00,.80,.01)
    epc=st.number_input("Equity put/call",.10,3.00,.60,.01)

if not selected:
    st.stop()

symbols=list(dict.fromkeys(selected+MARKET+VOL+SECTORS))
with st.spinner("Loading market data..."):
    data=get_data(tuple(symbols))

raw={"5Y":w5,"10Y":w10,"20Y":w20,"FULL":wf}
sm=sum(raw.values()) or 1
weights={k:v/sm for k,v in raw.items()}

cr=cycle_risk(data)
vix,vratio,vstatus,vr=vol_snapshot(data)
br,bp=breadth_risk(data,selected)
sr=pc_risk(tpc,epc) if usepc else 50.0
market=.30*cr+.25*vr+.25*br+.20*sr

nyse=mcal.get_calendar("NYSE")
sched=nyse.schedule(start_date=f"{int(year)}-09-01", end_date=f"{int(year)}-09-30")
sessions=pd.DatetimeIndex(sched.index).tz_localize(None)

rows=[]
for i,dt in enumerate(sessions,1):
    ps=[]; rs=[]
    row={"Date":dt.date(),"TD":i}
    for s in selected:
        p,r=blended_seasonality(data[s],i,weights)
        row[s]=p
        if pd.notna(p):ps.append(p)
        if pd.notna(r):rs.append(r)
    seasonal=float(np.mean(ps)) if ps else np.nan
    avgret=float(np.mean(rs)) if rs else np.nan
    final=(sshare/100)*seasonal+(1-sshare/100)*market if pd.notna(seasonal) else np.nan
    row.update({"Seasonal":seasonal,"AvgReturn":avgret,"Market":market,"Final":final,"Risk":risk_label(final)})
    rows.append(row)
df=pd.DataFrame(rows)

top=df.loc[df["Final"].idxmax()]
c1,c2,c3,c4=st.columns(4)
c1.metric("Highest-risk date",top["Date"].strftime("%b %d"))
c2.metric("Final score",f'{top["Final"]:.1f}')
c3.metric("Market regime",f"{market:.1f}")
c4.metric("VIX","N/A" if pd.isna(vix) else f"{vix:.2f}")

if top["Risk"] in ("HIGH","VERY HIGH"):
    st.error(f'Quick Signal: {top["Date"].strftime("%b %d")} — {top["Final"]:.1f}/100 ({top["Risk"]})')
elif top["Risk"]=="MODERATE":
    st.warning(f'Quick Signal: {top["Date"].strftime("%b %d")} — {top["Final"]:.1f}/100 ({top["Risk"]})')
else:
    st.success(f'Quick Signal: {top["Date"].strftime("%b %d")} — {top["Final"]:.1f}/100 ({top["Risk"]})')

st.subheader("Market Regime")
m1,m2,m3,m4=st.columns(4)
m1.metric("Cycles",f"{cr:.1f}")
m2.metric("Volatility",f"{vr:.1f}")
m3.metric("Breadth",f"{br:.1f}")
m4.metric("Sentiment",f"{sr:.1f}")
st.caption(f"VIX structure: {vstatus} | VIX/VIX3M: {'N/A' if pd.isna(vratio) else f'{vratio:.3f}'}")
st.caption(f"Breadth above 20/50/200 DMA: {bp.get(20,np.nan):.1f}% / {bp.get(50,np.nan):.1f}% / {bp.get(200,np.nan):.1f}%")

st.subheader(f"September {int(year)} Risk Calendar")
lookup={r["Date"].day:r for _,r in df.iterrows()}
first=date(int(year),9,1).weekday()
html="""<style>.riskcal{display:grid;grid-template-columns:repeat(7,1fr);gap:8px}.rh{text-align:center;font-weight:700}.rc{min-height:108px;padding:8px;border:1px solid #ccc;border-radius:11px}.rd{font-size:18px;font-weight:700}.rs{font-size:11px}</style><div class='riskcal'>"""
for h in ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]: html+=f"<div class='rh'>{h}</div>"
for _ in range(first): html+="<div class='rc' style='opacity:.12'></div>"
for day in range(1,cal.monthrange(int(year),9)[1]+1):
    r=lookup.get(day)
    if r is None:
        html+=f"<div class='rc' style='background:#e5e7eb'><div class='rd'>{day}</div><div class='rs'>Closed</div></div>"
    else:
        html+=f"<div class='rc' style='background:{color(r['Final'])};color:#111'><div class='rd'>{day}</div><div class='rs'>TD {int(r['TD'])}<br>Seasonal {r['Seasonal']:.1f}%<br><b>Final {r['Final']:.1f}</b><br>{r['Risk']}</div></div>"
html+="</div>"
st.markdown(html,unsafe_allow_html=True)

st.subheader("Highest-Risk Sessions")
rank=df.sort_values("Final",ascending=False)
st.dataframe(rank[["Date","TD","Seasonal","AvgReturn","Market","Final","Risk"]].style.format({
    "Seasonal":"{:.1f}%","AvgReturn":"{:.3f}%","Market":"{:.1f}","Final":"{:.1f}"
}),use_container_width=True,hide_index=True)

st.subheader("Risk Curve")
fig,ax=plt.subplots(figsize=(11,4))
ax.plot(df["Date"],df["Seasonal"],marker="o",label="Seasonality")
ax.plot(df["Date"],df["Final"],marker="o",label="Final risk")
ax.axhline(50,linestyle="--",linewidth=1)
ax.grid(alpha=.25);ax.legend()
fig.autofmt_xdate()
st.pyplot(fig,use_container_width=True)

st.subheader("Per-Stock Down Probability")
st.dataframe(df[["Date","TD"]+selected].style.format({s:"{:.1f}%" for s in selected}),use_container_width=True,hide_index=True)

st.download_button(
    "Download CSV",
    df.to_csv(index=False).encode(),
    file_name=f"september_{int(year)}_market_risk.csv",
    mime="text/csv"
)

st.info("Research/education only. Historical seasonality and model scores do not guarantee future returns.")
