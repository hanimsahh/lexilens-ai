import subprocess, sys

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "--quiet"])

for pkg in ["joblib", "pillow", "numpy",
            "anthropic", "python-dotenv", "matplotlib", "fpdf2", "pandas"]:
    try:
        __import__(pkg.replace("-","_").replace("fpdf2","fpdf").replace("pillow","PIL"))
    except ImportError:
        install(pkg)

import streamlit as st
import numpy as np
import joblib
import datetime
import os
import sqlite3
import time
import hashlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from dotenv import load_dotenv
import anthropic
from fpdf import FPDF

load_dotenv()

# ══════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="LexiLens AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════
# CSS — LIGHTER, WARMER THEME
# ══════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,400&family=JetBrains+Mono:wght@400;500&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@600;700&display=swap');

:root {
    --bg:      #f8f7f4;
    --card:    #ffffff;
    --hover:   #f3f2ef;
    --border:  rgba(0,0,0,0.07);
    --t1:      #1a1a1a;
    --t2:      #666560;
    --t3:      #999690;
    --ac:      #4f46e5;
    --ac2:     #6366f1;
    --ac-s:    rgba(26,92,53,0.08);
    --gold:    #d4a017;
    --amber:   #e08020;
    --purple:  #6b3fa0;
    --gr: #10b981;
    --am: #f59e0b;
    --pu: #8b5cf6;
    --shadow:  0 2px 8px rgba(0,0,0,0.06), 0 0 1px rgba(0,0,0,0.06);
    --shadow2: 0 8px 24px rgba(0,0,0,0.1), 0 0 1px rgba(0,0,0,0.06);
    --r:       14px;
    --r2:      10px;
}

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
    color: var(--t1);
    -webkit-font-smoothing: antialiased;
}

.stApp {
    background: var(--bg);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#4338ca 0%, #312e81 100%) !important;
    border-right: none !important;
    box-shadow: 4px 0 20px rgba(0,0,0,0.12);
}

[data-testid="stSidebar"] > div {
    padding: 0.2rem !important;
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] label {
    color: rgba(255,255,255,0.85) !important;
}

[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.15) !important;
    margin: 4px 0 !important;
}

[data-testid="stSidebar"] .stRadio > div {
    gap: 0px !important;
}

[data-testid="stSidebar"] .stRadio label {
    background: transparent !important;
    border: none !important;
    border-radius: var(--r2) !important;
    color: rgba(255,255,255,0.75) !important;
    font-size: 11px !important;
    font-weight: 500 !important;
    padding: 5px 10px !important;
    letter-spacing: 0 !important;
    text-transform: none !important;
    transition: all .15s !important;
    width: 100% !important;
}

[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(255,255,255,0.12) !important;
    color: #fff !important;
}

[data-testid="stSidebar"] .stRadio label[data-baseweb] {
    background: rgba(255,255,255,0.18) !important;
    color: #fff !important;
    font-weight: 600 !important;
}

[data-testid="stSidebar"] .stSelectbox > div > div {
    background: rgba(255,255,255,0.12) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: var(--r2) !important;
    color: #fff !important;
}

.stButton > button {
    background: linear-gradient(135deg,#4f46e5,#7c3aed) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    height: 52px !important;
    font-weight: 600 !important;
    font-size: 16px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 10px 25px rgba(99,102,241,0.25) !important;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 14px 30px rgba(99,102,241,0.35) !important;
}

[data-testid="stAppViewContainer"] {
    background:
    radial-gradient(circle at top left, rgba(124,58,237,0.08), transparent 30%),
    radial-gradient(circle at top right, rgba(99,102,241,0.08), transparent 30%),
    #f8f8fb;
}

[data-testid="stSidebar"] .stDownloadButton > button {
    background: rgba(255,255,255,0.18) !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    color: #fff !important;
    border-radius: var(--r2) !important;
    font-size: 12px !important;
}

.wm {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 14px;
    font-weight: 800;
    color: #fff !important;
    letter-spacing: -0.5px;
}

.ver {
    font-family: 'JetBrains Mono', monospace;
    font-size: 6px;
    color: rgba(255,255,255,0.45) !important;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 2px;
}

.ptitle {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 42px;
    font-weight: 800;
    color: var(--t1);
    letter-spacing: -1.5px;
    line-height: 1;
}

.psub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: var(--t3);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 6px;
}

.ey {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: var(--t3);
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.ey::before {
    content: '';
    display: inline-block;
    width: 14px;
    height: 2px;
    background: var(--ac);
    border-radius: 2px;
}

.card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--r);
    padding: 22px 24px;
    margin-bottom: 8px;
    box-shadow: var(--shadow);
    transition: all .2s;
}

.card:hover {
    box-shadow: var(--shadow2);
    transform: translateY(-1px);
}

.rc {
    background: linear-gradient(135deg, var(--ac) 0%, var(--ac2) 100%);
    border-radius: var(--r);
    padding: 32px 28px;
    text-align: center;
    box-shadow: 0 8px 30px rgba(99,102,241,0.30);
    margin-bottom: 16px;
    position: relative;
    overflow: hidden;
}

.rc::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 120px; height: 120px;
    border-radius: 50%;
    background: rgba(255,255,255,0.08);
}

.rc::after {
    content: '';
    position: absolute;
    bottom: -30px; left: -20px;
    width: 80px; height: 80px;
    border-radius: 50%;
    background: rgba(255,255,255,0.05);
}

.sc {
    position:relative;
    overflow:hidden;
    background:linear-gradient(180deg,rgba(255,255,255,0.96),rgba(248,250,252,0.92));
    border:1px solid rgba(99,102,241,0.08);
    border-radius:22px;
    padding:24px 20px;
    text-align:center;
    box-shadow:0 10px 30px rgba(15,23,42,0.06);
    transition:all .25s ease;
    backdrop-filter:blur(14px);
}

.sc::before{
    content:"";
    position:absolute;
    top:0; left:0;
    width:100%; height:4px;
    background:linear-gradient(90deg,#4f46e5,#6366f1);
    opacity:0.9;
}

.sc:hover{
    transform:translateY(-6px);
    box-shadow:0 20px 50px rgba(99,102,241,0.14);
    border-color:rgba(99,102,241,0.18);
}

.sn {
    font-family:'Plus Jakarta Sans',sans-serif;
    font-size:48px;
    font-weight:800;
    letter-spacing:-2px;
    line-height:1;
    background:linear-gradient(180deg,#111827,#374151);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}

@keyframes fadeUp{
    from{opacity:0;transform:translateY(8px);}
    to{opacity:1;transform:translateY(0);}
}

.sl {
    font-family: 'JetBrains Mono', monospace;
    font-size: 6px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--t3);
    margin-top: 6px;
}

.badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 5px 14px;
    border-radius: 99px;
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.3px;
}

.bn2 { background: rgba(99,102,241,0.10); color: #4f46e5; border: 1.5px solid rgba(99,102,241,0.20); }
.br2 { background: rgba(212,160,23,0.1); color: #a07010; border: 1.5px solid rgba(212,160,23,0.25); }
.bc2 { background: rgba(107,63,160,0.1); color: #6b3fa0; border: 1.5px solid rgba(107,63,160,0.2); }
.bhi { background: rgba(99,102,241,0.10); color: #4f46e5; border: 1.5px solid rgba(99,102,241,0.20); }
.blo { background: rgba(192,57,43,0.1); color: #c0392b; border: 1.5px solid rgba(192,57,43,0.2); }

.mr {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: var(--hover);
    border: 1px solid var(--border);
    border-radius: var(--r2);
    padding: 12px 16px;
    margin-bottom: 8px;
    transition: all .15s;
}

.mr:hover { background: var(--card); box-shadow: var(--shadow); }
.mn { font-size: 12px; color: var(--t2); font-weight: 500; }
.mv { font-family: 'JetBrains Mono', monospace; font-size: 14px; font-weight: 500; color: var(--t1); }

.pr { margin-bottom: 8px; }
.ph { display: flex; justify-content: space-between; margin-bottom: 6px; font-size: 12px; font-weight: 500; color: var(--t2); }
.pp { font-family: 'JetBrains Mono', monospace; color: var(--t1); font-weight: 500; }
.bt { height: 6px; background: rgba(0,0,0,0.06); border-radius: 99px; overflow: hidden; }
.bfn { height: 100%; border-radius: 99px; background: linear-gradient(90deg, #4f46e5, #818cf8); }
.bfr { height: 100%; border-radius: 99px; background: linear-gradient(90deg, #a07010, #d4a017); }
.bfc { height: 100%; border-radius: 99px; background: linear-gradient(90deg, #6b3fa0, #9b6dd4); }

.ex {
    background: var(--ac-s);
    border: 1px solid rgba(26,92,53,0.12);
    border-left: 3px solid var(--ac);
    border-radius: 0 var(--r2) var(--r2) 0;
    padding: 16px 18px;
    margin-top: 14px;
    font-size: 13px;
    color: var(--t2);
    line-height: 1.7;
}

.hi {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 12px;
    border-radius: var(--r2);
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.1);
    margin-bottom: 6px;
    font-size: 12px;
    transition: all .15s;
}

.hi:hover { background: rgba(255,255,255,0.14); border-color: rgba(255,255,255,0.2); }

.cu{
    background:linear-gradient(135deg,#4f46e5,#6366f1);
    border-radius:22px 22px 6px 22px;
    padding:18px 22px;
    margin-bottom:10px;
    font-size:14px;
    color:white;
    max-width:65%;
    margin-left:auto;
    line-height:1.8;
    box-shadow:0 12px 30px rgba(99,102,241,0.22);
    border:1px solid rgba(255,255,255,0.08);
    backdrop-filter:blur(10px);
    animation:fadeUp .25s ease;
}

.cb{
    background:rgba(255,255,255,0.92);
    border:1px solid rgba(99,102,241,0.08);
    border-radius:22px 22px 22px 6px;
    padding:18px 22px;
    margin-bottom:10px;
    font-size:14px;
    color:#475569;
    max-width:72%;
    line-height:1.85;
    box-shadow:0 10px 28px rgba(15,23,42,0.05);
    backdrop-filter:blur(12px);
    animation:fadeUp .25s ease;
}

.cl{
    font-family:'JetBrains Mono', monospace;
    font-size:10px !important;
    font-weight:800 !important;
    letter-spacing:1.5px !important;
    text-transform:uppercase;
    margin-bottom:10px;
    display:block;
}
.clu{ color:rgba(255,255,255,0.82) !important; }
.clb{ color:#7c3aed !important; }

.emp {
    background: var(--card);
    border: 1.5px dashed rgba(99,102,241,0.20);
    border-radius: var(--r);
    padding: 22px 16px;
    text-align: center;
    color: var(--t3);
    font-size: 13px;
    line-height: 1.7;
    box-shadow: var(--shadow);
}
.emp .ic { font-size: 32px; display: block; margin-bottom: 12px; opacity: 0.6; }

.alr { background: rgba(212,160,23,0.06); border: 1px solid rgba(212,160,23,0.2); border-left: 3px solid var(--gold); border-radius: 0 var(--r2) var(--r2) 0; padding: 14px 16px; margin-top: 12px; font-size: 13px; color: #6b4800; line-height: 1.6; }
.aln { background: var(--ac-s); border: 1px solid rgba(99,102,241,0.15); border-left: 3px solid var(--ac); border-radius: 0 var(--r2) var(--r2) 0; padding: 14px 16px; margin-top: 12px; font-size: 13px; color: #312e81; line-height: 1.6; }

.tp {
    background: linear-gradient(135deg, var(--ac) 0%, #312e81 100%);
    border-radius: var(--r);
    padding: 36px;
    text-align: center;
    margin-bottom: 16px;
    box-shadow: 0 8px 30px rgba(99,102,241,0.30);
    position: relative;
    overflow: hidden;
}

.tp::before {
    content: '';
    position: absolute;
    top: -50px; right: -30px;
    width: 140px; height: 140px;
    border-radius: 50%;
    background: rgba(255,255,255,0.06);
}

.tchar {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 100px;
    font-weight: 800;
    color: #fff;
    line-height: 1;
    margin: 12px 0 14px;
    letter-spacing: -3px;
}

.thint { font-size: 13px; color: rgba(255,255,255,0.65); line-height: 1.6; max-width: 260px; margin: 0 auto; font-weight: 400; }

.dots { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 10px; }

.dot {
    width: 32px; height: 32px; border-radius: 50%;
    border: 1.5px solid rgba(255,255,255,0.15);
    display: flex; align-items: center; justify-content: center;
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 12px; font-weight: 700;
    color: rgba(255,255,255,0.35);
    background: transparent;
    transition: all .2s;
}

.dot-active { background: #fff; color: var(--ac); border-color: #fff; box-shadow: 0 4px 12px rgba(255,255,255,0.3); }
.dot-done { background: rgba(255,255,255,0.15); color: rgba(255,255,255,0.8); border-color: rgba(255,255,255,0.25); }

.user-pill { display: flex; align-items: center; gap: 10px; background: rgba(255,255,255,0.12); border: 1px solid rgba(255,255,255,0.15); border-radius: var(--r); padding: 6px 8px; margin-bottom: 6px; }
.user-avatar { width: 32px; height: 32px; border-radius: 50%; background: rgba(255,255,255,0.2); display: flex; align-items: center; justify-content: center; font-size: 14px; font-weight: 800; color: #fff; }
.user-name { font-size: 13px; font-weight: 700; color: #fff !important; }
.user-role { font-size: 10px; color: rgba(255,255,255,0.5) !important; font-family: 'JetBrains Mono', monospace; letter-spacing: 1px; }

div[data-testid="stFileUploader"]{
    min-height:180px !important;
    border-radius:30px !important;
    border:2px dashed rgba(99,102,241,0.30) !important;
    background:linear-gradient(180deg,rgba(255,255,255,0.98),rgba(248,250,252,0.95)) !important;
    display:flex !important;
    align-items:center !important;
    justify-content:center !important;
    box-shadow:0 20px 60px rgba(99,102,241,0.08) !important;
    transition:all .25s ease !important;
    position:relative !important;
}

div[data-testid="stFileUploader"]:hover{
    transform:translateY(-3px);
    border-color:rgba(99,102,241,0.55) !important;
    box-shadow:0 30px 80px rgba(99,102,241,0.16) !important;
}

div[data-testid="stFileUploader"] section{
    padding-top:40px !important;
    text-align:center !important;
    background:transparent !important;
}

div[data-testid="stFileUploader"] > section > div {
    background:transparent !important;
    border:none !important;
    box-shadow:none !important;
}

div[data-testid="stFileUploader"]::before{
    content:"☁️";
    position:absolute;
    top:42px; left:50%;
    transform:translateX(-50%);
    font-size:34px;
}

div[data-testid="stFileUploader"] button{
    background:linear-gradient(135deg,#4f46e5,#6366f1) !important;
    color:white !important;
    border:none !important;
    border-radius:14px !important;
    height:46px !important;
    padding:0 24px !important;
    font-weight:600 !important;
    box-shadow:0 12px 30px rgba(99,102,241,0.25) !important;
}

.stButton > button {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    border-radius: var(--r2) !important;
    transition: all .2s !important;
    letter-spacing: -0.1px !important;
    padding: 6px 12px !important;
}

.stButton > button[kind="primary"]:hover {
    background: var(--ac2) !important;
    box-shadow: 0 6px 20px rgba(99,102,241,0.40) !important;
    transform: translateY(-1px) !important;
}

.stButton > button:not([kind="primary"]) {
    background: var(--card) !important;
    border: 1.5px solid var(--border) !important;
    color: var(--t1) !important;
    box-shadow: var(--shadow) !important;
}

.stButton > button:not([kind="primary"]):hover {
    border-color: rgba(99,102,241,0.30) !important;
    color: var(--ac) !important;
    transform: translateY(-1px) !important;
}

.stTextInput > div > div > input {
    background: var(--card) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: var(--r2) !important;
    color: var(--t1) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 14px !important;
    font-weight: 400 !important;
    padding: 12px 16px !important;
    box-shadow: var(--shadow) !important;
    transition: all .2s !important;
}

.stTextInput > div > div > input:focus {
    border-color: var(--ac) !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.10) !important;
}

.stTextInput label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 10px !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    color: var(--t3) !important;
    font-weight: 500 !important;
}

.stRadio label {
    font-size: 11px !important;
    font-weight: 500 !important;
    color: var(--t2) !important;
    padding: 9px 14px !important;
    border-radius: var(--r2) !important;
    border: 1.5px solid var(--border) !important;
    background: var(--card) !important;
    transition: all .15s !important;
    box-shadow: var(--shadow) !important;
}

.stRadio label:hover {
    border-color: rgba(99,102,241,0.30) !important;
    color: var(--ac) !important;
}

.stSelectbox > div > div {
    background: var(--card) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: var(--r2) !important;
    box-shadow: var(--shadow) !important;
}

.feature-card{
    background:rgba(255,255,255,0.72);
    backdrop-filter:blur(14px);
    border:1px solid rgba(99,102,241,0.08);
    border-radius:24px;
    padding:28px;
    transition:all .25s ease;
}

.feature-card:hover{
    transform:translateY(-8px);
    box-shadow:0 24px 60px rgba(99,102,241,0.14);
    border-color:rgba(99,102,241,0.18);
}

button[kind="primary"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 14px 30px rgba(92,95,246,0.38) !important;
}

.stDataFrame { border-radius: var(--r) !important; overflow: hidden; box-shadow: var(--shadow); }

html, body, [data-testid="stAppViewContainer"] {
    overflow-x: hidden !important;
}

.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 0rem !important;
    max-width: 1200px !important;
}

section.main > div {
    padding-top: 0rem !important;
}

[data-testid="stVerticalBlock"] {
    gap: 0.6rem !important;
}

.stForm {
    background: rgba(255,255,255,0.75) !important;
    border: 1px solid rgba(99,102,241,0.10) !important;
    border-radius: 24px !important;
    padding: 24px !important;
    backdrop-filter: blur(12px);
    box-shadow: 0 10px 30px rgba(99,102,241,0.08);
}

header[data-testid="stHeader"]{ display:none; }
[data-testid="stToolbar"]{ display:none; }

.block-container{
    padding-top:0rem !important;
    padding-bottom:0rem !important;
}

hr { border-color: var(--border) !important; margin: 16px 0 !important; }
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-thumb { background: var(--ac); border-radius: 99px; }

[data-testid="stSidebar"] .stButton > button {
    background: rgba(255,255,255,0.10) !important;
    border: 1px solid rgba(255,255,255,0.20) !important;
    color: rgba(255,255,255,0.85) !important;
    border-radius: 10px !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    width: 100% !important;
    box-shadow: none !important;
}

[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(255,255,255,0.18) !important;
    color: #fff !important;
    transform: none !important;
    box-shadow: none !important;
}

</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# DATABASE
# ══════════════════════════════════════════════════════════
def init_db():
    conn = sqlite3.connect("lexilens.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT,
            created_at TEXT,
            role TEXT DEFAULT 'user'
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT, label TEXT, confidence REAL,
            source TEXT, age_group TEXT, username TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS test_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT, char TEXT, label TEXT,
            confidence REAL, response_time REAL,
            age_group TEXT, username TEXT
        )
    """)
    conn.commit()
    conn.close()

def hash_password(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

def register_user(username, password):
    try:
        conn = sqlite3.connect("lexilens.db")
        c = conn.cursor()
        c.execute("INSERT INTO users (username,password,created_at) VALUES (?,?,?)",
                  (username, hash_password(password), datetime.datetime.now().isoformat()))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False

def login_user(username, password):
    conn = sqlite3.connect("lexilens.db")
    c = conn.cursor()
    c.execute("SELECT username FROM users WHERE username=? AND password=?",
              (username, hash_password(password)))
    user = c.fetchone()
    conn.close()
    return user[0] if user else None

def save_prediction(label, conf, source, age, username):
    try:
        conn = sqlite3.connect("lexilens.db")
        c = conn.cursor()
        c.execute("INSERT INTO predictions VALUES (NULL,?,?,?,?,?,?)",
                  (datetime.datetime.now().isoformat(), label, conf, source, age, username))
        conn.commit()
        conn.close()
    except Exception as e:
        st.error(str(e))

def load_user_history(username):
    conn = sqlite3.connect("lexilens.db")
    cursor = conn.cursor()

    cursor.execute("""
        SELECT label, confidence, source, timestamp
        FROM predictions
        WHERE username = ?
        ORDER BY id DESC
        LIMIT 20
    """, (username,))

    rows = cursor.fetchall()
    conn.close()

    history = []

    for row in rows:
        history.append({
            "label": row[0],
            "confidence": row[1],
            "source": row[2],
            "time": row[3]
        })

    return history


def save_test_result(char, label, conf, rt, age, username):
    try:
        conn = sqlite3.connect("lexilens.db")
        c = conn.cursor()
        c.execute("INSERT INTO test_results VALUES (NULL,?,?,?,?,?,?,?)",
                  (datetime.datetime.now().isoformat(), char, label, conf, rt, age, username))
        conn.commit()
        conn.close()
    except Exception as e:
        st.error(str(e))

init_db()

# ══════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════
LABEL_MAP    = {0:"Corrected", 1:"Normal", 2:"Reversal"}
LABEL_COLORS = {"Normal":"#4f46e5","Reversal":"#a07010","Corrected":"#6b3fa0"}
LABEL_BADGE  = {"Normal":"bn2","Reversal":"br2","Corrected":"bc2"}
LABEL_BAR    = {"Normal":"bfn","Reversal":"bfr","Corrected":"bfc"}
DISPLAY      = ["Normal","Reversal","Corrected"]

TEST_SEQ = [
    {"char":"b","hint":"Write the letter b naturally"},
    {"char":"d","hint":"Write the letter d naturally"},
    {"char":"p","hint":"Write the letter p naturally"},
    {"char":"q","hint":"Write the letter q naturally"},
    {"char":"6","hint":"Write the number 6 naturally"},
    {"char":"9","hint":"Write the number 9 naturally"},
    {"char":"3","hint":"Write the number 3 naturally"},
    {"char":"7","hint":"Write the number 7 naturally"},
]

AGE_CONTEXT = {
    "Child (5-12)": {
        "Normal":    "For a child aged 5-12, this is a correctly oriented character. No concerns identified.",
        "Reversal":  "Occasional reversals are normal up to age 7. If older than 7-8 and reversals persist, consider monitoring and discussing with a teacher.",
        "Corrected": "Self-correction is a positive sign — the child is aware of correct letter formation.",
    },
    "Teen (13-17)": {
        "Normal":    "Character written correctly. No reversal pattern detected.",
        "Reversal":  "Persistent reversals in teenagers may indicate an undiagnosed specific learning difficulty. A formal assessment is recommended.",
        "Corrected": "Self-correction suggests awareness of reversal tendency. Consider discussing with a learning support specialist.",
    },
    "Adult (18+)": {
        "Normal":    "Character written correctly. No reversal pattern detected.",
        "Reversal":  "Reversal patterns in adults may indicate dyslexia not identified in childhood. Consider a formal assessment.",
        "Corrected": "Self-correction detected. This may reflect a long-standing strategy to compensate for reversal tendencies.",
    },
}

EXPLANATIONS = {
    "Normal":    "✅ <b>Normal Pattern</b> — The character is written in the expected orientation. No reversal detected.",
    "Reversal":  "⚠️ <b>Reversal Pattern</b> — The character appears mirrored or rotated, a common indicator associated with dyslexia.",
    "Corrected": "🔄 <b>Corrected Pattern</b> — Evidence of self-correction during writing. The character was started incorrectly then adjusted.",
}

SYSTEM_PROMPT = """
You are LexiLens Assistant, a supportive and ethically responsible AI educational assistant specialising in dyslexia, handwriting analysis, and the LexiLens AI platform.

Your role is to help parents, teachers, caregivers, and students better understand handwriting reversal patterns and dyslexia-related indicators in a calm, accessible, and non-judgemental way.

Core Responsibilities:
- Explain dyslexia and how it may affect handwriting, letter orientation, and character formation
- Interpret LexiLens AI prediction results using simple and accessible language
- Provide educational guidance and supportive next steps when reversal patterns are detected
- Help users understand confidence scores, uncertainty, and possible limitations of AI predictions
- Encourage awareness and educational support rather than fear or alarm

Communication Style:
- Warm, supportive, professional, and reassuring
- Calm and non-alarmist
- Clear and easy for non-specialists to understand
- Concise and conversational
- Prefer short paragraphs or bullet points where useful
- Avoid overly academic or technical explanations unless specifically requested

Ethical and Safety Rules:
- Never claim that a user definitely has dyslexia
- Never provide medical or clinical diagnoses
- Never use definitive or alarming statements
- Avoid phrases such as:
  * "This confirms dyslexia"
  * "The user has dyslexia"
  * "This is a definite indicator"
  * "This proves a learning disorder"

Instead, use cautious and responsible language such as:
- "may indicate"
- "could be associated with"
- "may suggest"
- "can sometimes be linked to"
- "may benefit from further educational observation"

When discussing reversal patterns:
- Clarify that occasional reversals can be developmentally normal in younger children
- Mention that persistent reversal patterns may sometimes be associated with dyslexia-related traits
- Encourage professional educational assessment when appropriate
- Emphasise that AI predictions contain uncertainty and should be interpreted carefully

Confidence and Uncertainty Handling:
- If prediction confidence is low, clearly acknowledge uncertainty
- Encourage users to upload clearer handwriting samples when appropriate
- Avoid sounding overly certain about borderline predictions
- Present predictions as supportive indicators rather than conclusions

System Scope:
- Only answer questions related to:
  * dyslexia
  * handwriting analysis
  * educational screening
  * handwriting reversals
  * the LexiLens AI platform

Important Disclaimer:
LexiLens AI is an educational screening support tool designed for research and educational purposes only.
It is not a medical device and should not replace professional educational or clinical assessment.
"""

# ══════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════
@st.cache_resource
def load_models():
    r = {"clf":None,"scaler":None,"le":None}
    for k,f in [("clf","dyslexia_model.pkl"),("scaler","scaler.pkl"),("le","label_encoder.pkl")]:
        try: r[k] = joblib.load(f)
        except Exception as e:
            st.error(str(e))
    return r

M = load_models()

# ══════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════
DEFAULTS = {
    "logged_in":    False,
    "username":     None,
    "auth_page":    "welcome",
    "history":      [],
    "last_result":  None,
    "chat_messages":[],
    "chat_ctx":     "",
    "test_step":    0,
    "test_results": [],
    "test_phase":   "idle",
    "age_group":    "Child (5-12)",
    "current_page": "🏠 Dashboard",
}
for k,v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════
# CORE HELPERS
# ══════════════════════════════════════════════════════════
def preprocess(img_pil):
    img = img_pil.convert("L").resize((28,28))
    arr = np.array(img, dtype=np.float32)
    if arr.max() > 0: arr = arr / arr.max()
    return arr.flatten()

def predict(flat):
    try:
        x = flat.reshape(1,-1)
        if M["scaler"]: x = M["scaler"].transform(x)
        raw = M["clf"].predict_proba(x)[0] if hasattr(M["clf"],"predict_proba") else np.eye(3)[M["clf"].predict(x)[0]]
        idx   = int(np.argmax(raw))
        label = LABEL_MAP[idx]
        conf  = float(raw[idx])
        probs = {"Corrected":float(raw[0]),"Normal":float(raw[1]),"Reversal":float(raw[2])}
        return label, conf, probs
    except:
        return "Normal", 0.60, {"Normal":0.60,"Reversal":0.25,"Corrected":0.15}

def add_history(label, conf, source):
    st.session_state.history.append({
        "label":label,"confidence":conf,"source":source,
        "time":datetime.datetime.now().strftime("%H:%M:%S"),
    })
    save_prediction(label, conf, source,
                    st.session_state.get("age_group","Unknown"),
                    st.session_state.get("username","guest"))

def get_ai_response(msg):
    key = os.getenv("api_key") or os.getenv("ANTHROPIC_API_KEY")
    if not key: return "⚠️ API key not found in .env file."
    try:
        client = anthropic.Anthropic(api_key=key)
        msgs = [{"role":m["role"],"content":m["content"]}
                for m in st.session_state.chat_messages[-10:]]
        msgs.append({"role":"user","content":msg})
        sys_p = SYSTEM_PROMPT
        if st.session_state.chat_ctx:
            sys_p += f"\n\nSession context: {st.session_state.chat_ctx}"
        r = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=350,
            system=sys_p + """

        You are a friendly AI dyslexia screening assistant inside the LexiLens AI platform.

        Rules:
        - Keep responses concise and conversational.
        - Avoid long academic paragraphs.
        - Use short paragraphs.
        - Be warm, supportive, and interactive.
        - Maximum 120 words unless the user asks for detail.
        - Explain concepts simply for parents and educators.
        - Occasionally ask follow-up questions.
        - Use bullet points when helpful.
        """,
            messages=msgs
        )
        return r.content[0].text
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

def generate_pdf():
    pdf = FPDF()
    pdf.add_page()
    
    user = st.session_state.get("username", "Guest")  # ← EN BAŞA AL
    h = st.session_state.history
    
    pdf.set_fill_color(79,70,229)
    pdf.rect(0,0,210,40,'F')
    pdf.set_font("Helvetica","B",20)
    pdf.set_text_color(255,255,255)
    pdf.set_xy(10,12)
    pdf.cell(0,10,"LexiLens AI - Session Report", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica","",9)
    pdf.set_text_color(200,200,220)
    pdf.set_xy(10,26)
    pdf.cell(0,6,f"User: {user}  |  Generated: {datetime.datetime.now().strftime('%d %B %Y, %H:%M')}", new_x="LMARGIN", new_y="NEXT")
    pdf.set_y(50)
    h = st.session_state.history
    if not h:
        pdf.set_font("Helvetica","",11)
        pdf.set_text_color(100,100,120)
        pdf.cell(0,10,"No predictions recorded in this session.",ln=True)
    else:
        labels = [x["label"] for x in h]
        n_rev  = labels.count("Reversal")
        total  = len(labels)
        pdf.set_fill_color(245,243,255)
        pdf.set_draw_color(79,70,229)
        pdf.rect(10,pdf.get_y(),190,36,'FD')
        pdf.set_font("Helvetica","B",9)
        pdf.set_text_color(79,70,229)
        pdf.set_xy(14,pdf.get_y()+4)
        pdf.cell(0,5,"SESSION SUMMARY",ln=True)
        pdf.set_font("Helvetica","",10)
        pdf.set_text_color(60,60,80)
        pdf.set_x(14)
        pdf.cell(60,6,f"Total: {total}")
        pdf.cell(60,6,f"Reversals: {n_rev}")
        pdf.cell(60,6,f"Rate: {n_rev/total*100:.1f}%",ln=True)
        pdf.ln(8)
        cm = {"Normal":(5,150,105),"Reversal":(217,119,6),"Corrected":(124,58,237)}
        pdf.set_font("Helvetica","B",9)
        pdf.set_text_color(79,70,229)
        pdf.set_x(10)
        for hdr,w in [("#",10),("Time",28),("Source",40),("Prediction",55),("Confidence",55)]:
            pdf.set_fill_color(245,243,255)
            pdf.cell(w,8,hdr,border=1,fill=True,align="C")
        pdf.ln()
        for i,item in enumerate(h,1):
            pdf.set_font("Helvetica","",9)
            if i%2==0:
                pdf.set_fill_color(250,249,255)
            else:
                pdf.set_fill_color(255,255,255)
            pdf.set_text_color(60,60,80)
            pdf.set_x(10)
            pdf.cell(10,7,str(i),border=1,fill=True,align="C")
            pdf.cell(28,7,item["time"],border=1,fill=True,align="C")
            pdf.cell(40,7,item["source"][:18],border=1,fill=True,align="C")
            r,g,b = cm.get(item["label"],(60,60,80))
            pdf.set_text_color(r,g,b)
            pdf.cell(55,7,item["label"],border=1,fill=True,align="C")
            pdf.set_text_color(60,60,80)
            pdf.cell(55,7,f"{item['confidence']*100:.1f}%",border=1,fill=True,align="C")
            pdf.ln()
    pdf.set_y(-20)
    pdf.set_font("Helvetica","",8)
    pdf.set_text_color(160,160,170)
    pdf.cell(0,5,"LexiLens AI - For educational and research purposes only. Not a clinical diagnostic tool.",align="C")
    output = pdf.output(dest='S')
    if isinstance(output, bytes):
        return output
    return bytes(pdf.output())

def send_pdf_email(to_email, pdf_bytes, username):
    import requests
    key = os.getenv("RESEND_API_KEY")
    if not key:
        return False, "Resend API key not found."
    try:
        import base64
        pdf_b64 = base64.b64encode(pdf_bytes).decode()
        response = requests.post(
            "https://api.resend.com/emails",
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json"
            },
            json={
                "from": "LexiLens AI <onboarding@resend.dev>",
                "to": [to_email],
                "subject": "Your LexiLens AI Screening Report",
                "html": f"""
                <div style="font-family:sans-serif;max-width:600px;margin:0 auto;">
                    <div style="background:linear-gradient(135deg,#4f46e5,#7c3aed);
                                padding:32px;border-radius:16px 16px 0 0;text-align:center;">
                        <h1 style="color:white;margin:0;font-size:28px;">🧠 LexiLens AI</h1>
                        <p style="color:rgba(255,255,255,0.8);margin-top:8px;">Handwriting Pattern Analysis Report</p>
                    </div>
                    <div style="background:#f8f8fb;padding:32px;border-radius:0 0 16px 16px;">
                        <p style="color:#1a1a1a;font-size:16px;">Hi <b>{username}</b>,</p>
                        <p style="color:#6b7280;line-height:1.7;">
                            Please find your LexiLens AI screening session report attached.
                            This report contains your handwriting pattern analysis results.
                        </p>
                        <div style="background:rgba(99,102,241,0.06);border-left:4px solid #6366f1;
                                    padding:16px;border-radius:0 8px 8px 0;margin:20px 0;">
                            <p style="color:#4f46e5;font-weight:700;margin:0 0 6px;">⚠️ Important Notice</p>
                            <p style="color:#64748b;font-size:13px;margin:0;line-height:1.6;">
                                This report is for educational purposes only and does not constitute
                                a clinical diagnosis. Please consult a qualified professional.
                            </p>
                        </div>
                        <p style="color:#9ca3af;font-size:12px;margin-top:24px;">
                            Generated by LexiLens AI · University of Huddersfield FYP
                        </p>
                    </div>
                </div>
                """,
                "attachments": [{
                    "filename": f"lexilens_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    "content": pdf_b64
                }]
            }
        )
        if response.status_code == 200:
            return True, "Email sent!"
        else:
            return False, f"Error: {response.json()}"
    except Exception as e:
        return False, str(e)

def render_result(label, conf, probs):
    color = LABEL_COLORS[label]
    badge = LABEL_BADGE[label]
    risk  = (1-conf)*100
    cb    = "bhi" if conf>=0.7 else "blo"

    st.markdown(f"""
    <div class="rc">
        <div style="font-family:'JetBrains Mono',monospace;font-size:9px;letter-spacing:3px;
                    text-transform:uppercase;color:rgba(255,255,255,0.7);margin-bottom:10px;">
            Detection Result
        </div>
        <div style="font-family:'Fraunces',serif;font-size:40px;font-weight:700;
                    color:#fff;letter-spacing:-1px;margin:6px 0 12px;">{label}</div>
        <div class="badge {badge}" style="background:rgba(26,77,46,0.3);color:#c7d2fe;border-color:rgba(26,77,46,0.4);font-family:IBM Plex Mono,monospace;">
            {label.upper()} PATTERN
        </div>
    </div>
    <div class="mr">
        <div class="mn">Confidence</div>
        <div class="mv" style="color:{color};">{conf*100:.1f}%
            <span class="badge {cb}" style="margin-left:8px;font-size:9px;">
                {'HIGH' if conf>=0.7 else 'LOW'}</span></div></div>
    <div class="mr"><div class="mn">Uncertainty</div><div class="mv">{risk:.0f}%</div></div>
    <div class="mr"><div class="mn">Model</div>
        <div class="mv" style="font-size:11px;color:var(--t2);">Random Forest · 85.85%</div></div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="ey" style="margin-top:18px;">Class Probabilities</div>', unsafe_allow_html=True)
    for name in DISPLAY:
        pct = int(probs[name]*100)
        st.markdown(f"""
        <div class="pr">
            <div class="ph"><span>{name}</span><span class="pp">{pct}%</span></div>
            <div class="bt"><div class="{LABEL_BAR[name]}" style="width:{pct}%;"></div></div>
        </div>""", unsafe_allow_html=True)

    if label=="Reversal":
        st.markdown('<div class="alr">⚠️ <b>Reversal detected.</b> If persistent beyond age 7-8, consult an educational professional.</div>', unsafe_allow_html=True)
    elif label=="Normal":
        st.markdown('<div class="aln">✅ <b>No reversal detected.</b> The character appears correctly oriented.</div>', unsafe_allow_html=True)

    age = st.session_state.get("age_group","Child (5-12)")
    age_msg = AGE_CONTEXT.get(age,{}).get(label,"")
    if age_msg:
        st.markdown(f"""
        <div style='background:rgba(79,70,229,0.05);border:1px solid rgba(79,70,229,0.12);
                    border-left:3px solid #4f46e5;border-radius:0 10px 10px 0;
                    padding:12px 16px;margin-top:10px;font-size:12px;color:var(--t2);line-height:1.6;'>
            <div style='font-family:"JetBrains Mono",monospace;font-size:9px;letter-spacing:2px;
                        text-transform:uppercase;color:#4f46e5;margin-bottom:5px;'>{age} Context</div>
            {age_msg}
        </div>""", unsafe_allow_html=True)

    st.markdown(f'<div class="ex">{EXPLANATIONS[label]}</div>', unsafe_allow_html=True)

    if conf < 0.5:
        st.markdown('<div class="alr">⚠️ <b>Low confidence.</b> Please upload a clearer, well-lit photograph.</div>', unsafe_allow_html=True)
    elif conf < 0.7:
        st.markdown('<div style="background:rgba(217,119,6,0.05);border:1px solid rgba(217,119,6,0.15);border-left:3px solid #d97706;border-radius:0 10px 10px 0;padding:12px 16px;margin-top:10px;font-size:12px;color:#92400e;">ℹ️ <b>Moderate confidence.</b> Consider retesting with a clearer image.</div>', unsafe_allow_html=True)

    st.session_state.chat_ctx = (
        f"Last prediction: {label} with {conf*100:.1f}% confidence. "
        f"Probabilities: Normal={probs['Normal']*100:.0f}%, "
        f"Reversal={probs['Reversal']*100:.0f}%, Corrected={probs['Corrected']*100:.0f}%."
    )


# ══════════════════════════════════════════════════════════
# AUTH PAGES — ORTAK CSS
# ══════════════════════════════════════════════════════════

AUTH_CSS = """
<style>
/* ── Tüm auth sayfaları için genel reset ── */
header[data-testid="stHeader"]   { display: none !important; }
[data-testid="stToolbar"]        { display: none !important; }

/* Auth sayfalarında sidebar'ı gizle */
[data-testid="stSidebar"]        { display: none !important; }

.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

section.main > div {
    padding: 0 !important;
}

/* ── TAM EKRAN ARKA PLAN ── */
.auth-bg {
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 80% 60% at 15% -5%,  rgba(99,102,241,0.20) 0%, transparent 55%),
        radial-gradient(ellipse 60% 50% at 85% 105%, rgba(124,58,237,0.16) 0%, transparent 50%),
        #f4f3f8;
    z-index: 0;
    pointer-events: none;
}

/* dekoratif daireler */
.auth-bg::before {
    content: '';
    position: absolute;
    top: -200px; right: -200px;
    width: 550px; height: 550px;
    border-radius: 50%;
    background: rgba(99,102,241,0.07);
    filter: blur(70px);
}
.auth-bg::after {
    content: '';
    position: absolute;
    bottom: -160px; left: -140px;
    width: 450px; height: 450px;
    border-radius: 50%;
    background: rgba(124,58,237,0.06);
    filter: blur(60px);
}

/* ── İÇERİK WRAPPER ── */
.auth-center {
    position: fixed;
    inset: 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    z-index: 1;
    pointer-events: none;   /* sadece görsel — Streamlit widgetları üstte */
}

/* ── KART ── */
.auth-card {
    background: rgba(255,255,255,0.88);
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    border: 1px solid rgba(99,102,241,0.13);
    border-radius: 28px;
    padding: 44px 48px 36px;
    width: 100%;
    max-width: 440px;
    box-shadow:
        0 32px 80px rgba(99,102,241,0.13),
        0 0 0 1px rgba(255,255,255,0.65) inset;
    animation: authUp .45s cubic-bezier(.22,.68,0,1.15) both;
    pointer-events: auto;
}

@keyframes authUp {
    from { opacity:0; transform: translateY(30px) scale(0.97); }
    to   { opacity:1; transform: translateY(0)    scale(1);    }
}

/* ── LOGO ── */
.auth-logo {
    width: 56px; height: 56px;
    border-radius: 16px;
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    display: flex; align-items: center; justify-content: center;
    font-size: 26px; margin: 0 auto 18px;
    box-shadow: 0 10px 28px rgba(99,102,241,0.30);
}

.auth-title-txt {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 27px; font-weight: 800;
    color: #111827; letter-spacing: -0.8px;
    text-align: center; margin-bottom: 5px;
}
.auth-sub-txt {
    font-size: 13px; color: #9ca3af; font-weight: 400;
    text-align: center; margin-bottom: 0; line-height: 1.55;
}

/* ── FORM override: auth sayfalarında form kutu görünümü ── */
.auth-form-wrap [data-testid="stForm"] {
    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    padding: 0 !important;
    box-shadow: none !important;
}

/* ── WELCOME HERO ── */
.hero-brand {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 78px; font-weight: 900;
    letter-spacing: -4px; line-height: 0.9;
    color: #111827; text-align: center;
}
.hero-brand span {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-sub {
    font-size: 17px; line-height: 1.7; color: #6b7280;
    max-width: 460px; text-align: center;
    margin: 16px auto 0; font-weight: 400;
}

/* ── TAG PILL ── */
.tag-pill {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(99,102,241,0.09);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 99px; padding: 7px 18px;
    font-size: 12px; font-weight: 600; color: #4f46e5;
}

/* ── FEATURE BADGES ── */
.feat-row {
    display: flex; flex-wrap: wrap; justify-content: center; gap: 8px;
    margin: 20px 0 28px;
}
.feat-badge {
    background: rgba(99,102,241,0.07);
    border: 1px solid rgba(99,102,241,0.16);
    border-radius: 10px; padding: 7px 15px;
    font-size: 12px; font-weight: 600; color: #4f46e5;
}

/* ── STATS ŞERİDİ ── */
.stats-strip {
    display: flex; justify-content: center; gap: 32px;
    background: rgba(255,255,255,0.80);
    border: 1px solid rgba(99,102,241,0.12);
    border-radius: 16px; padding: 14px 28px;
    backdrop-filter: blur(8px);
}
.stat-num  { font-size: 24px; font-weight: 800; color: #4f46e5; letter-spacing: -1px; line-height:1; }
.stat-lbl  { font-size: 10px; color: #9ca3af; letter-spacing: 1.5px; text-transform: uppercase; margin-top: 3px; }
.stat-sep  { width: 1px; background: rgba(99,102,241,0.15); }

/* ── Streamlit buton konumlandırma (welcome) ── */
.welcome-btn-row {
    display: flex; justify-content: center; gap: 12px;
    margin: 28px auto 0; max-width: 340px;
}
</style>
"""

# ══════════════════════════════════════════════════════════
# WELCOME PAGE
# ══════════════════════════════════════════════════════════

def show_welcome():
    st.markdown(AUTH_CSS, unsafe_allow_html=True)
    st.markdown('<div class="auth-bg"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center;margin-bottom:22px;padding-top:12vh;">
        <div class="tag-pill">
            🧠 AI-Powered Educational Screening
        </div>
    </div>

    <div class="hero-brand">
        LexiLens<br><span>AI</span>
    </div>

    <div class="hero-sub">
        Handwriting pattern analysis platform supporting dyslexia screening
        through machine learning and AI-assisted interpretation.
    </div>

    <div class="feat-row">
        <div class="feat-badge">🎯 Guided Screening</div>
        <div class="feat-badge">🤖 AI Assistant</div>
        <div class="feat-badge">📊 Smart Insights</div>
        <div class="feat-badge">📄 PDF Reports</div>
        <div class="feat-badge">📂 Batch Analysis</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:4vh'></div>", unsafe_allow_html=True)

    col_l, col1, col2, col_r = st.columns([2, 1.2, 1.2, 2])
    with col1:
        if st.button("🚀 Get Started", type="primary", use_container_width=True):
            st.session_state.auth_page = "register"
            st.rerun()
    with col2:
        if st.button("Sign In →", use_container_width=True):
            st.session_state.auth_page = "login"
            st.rerun()

    st.markdown("""
    <div style="display:flex;justify-content:center;margin-top:20px;">
        <div class="stats-strip">
            <div style="text-align:center;">
                <div class="stat-num">85.85%</div>
                <div class="stat-lbl">Accuracy</div>
            </div>
            <div class="stat-sep"></div>
            <div style="text-align:center;">
                <div class="stat-num">151k</div>
                <div class="stat-lbl">Training Images</div>
            </div>
            <div class="stat-sep"></div>
            <div style="text-align:center;">
                <div class="stat-num">91.9</div>
                <div class="stat-lbl">SUS Score</div>
            </div>
            <div class="stat-sep"></div>
            <div style="text-align:center;">
                <div class="stat-num">3</div>
                <div class="stat-lbl">Pattern Classes</div>
            </div>
        </div>
    </div>
    <div style="text-align:center;margin-top:10px;font-size:11px;color:#c5c5d0;">
        For educational and research purposes only. Not a clinical diagnostic tool.
    </div>
    """, unsafe_allow_html=True)


def show_login():
    st.markdown(AUTH_CSS, unsafe_allow_html=True)
    st.markdown('<div class="auth-bg"></div>', unsafe_allow_html=True)

    st.markdown("""
    <style>
    [data-testid="stSidebar"] { display: none !important; }
    .block-container { padding: 0 !important; max-width: 100% !important; }
    section.main > div { padding: 0 !important; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="display:flex;flex-direction:column;align-items:center;padding-top:18vh;margin-bottom:10px;">
        <div style="width:52px;height:52px;border-radius:14px;background:linear-gradient(135deg,#4f46e5,#7c3aed);
                    display:flex;align-items:center;justify-content:center;font-size:24px;
                    box-shadow:0 10px 28px rgba(99,102,241,0.32);margin-bottom:14px;">🧠</div>
        <div style="font-family:'Plus Jakarta Sans',sans-serif;font-size:26px;font-weight:800;
                    color:#111827;letter-spacing:-0.7px;margin-bottom:4px;text-align:center;">Welcome back</div>
        <div style="font-size:13px;color:#9ca3af;margin-bottom:24px;text-align:center;">
            Sign in to continue to LexiLens AI</div>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_m, col_r = st.columns([1, 2, 1])
    with col_m:
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            submit   = st.form_submit_button("Sign In →", type="primary", use_container_width=True)

        if submit:
            if username and password:
                user = login_user(username, password)
                if user:
                    st.session_state.logged_in = True
                    st.session_state.username  = user
                    st.session_state.history = load_user_history(user)
                    st.session_state.auth_page = "app"
                    st.success(f"Welcome back, {user}!")
                    st.rerun()
                else:
                    st.error("Incorrect username or password.")
            else:
                st.warning("Please fill in all fields.")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("← Back", use_container_width=True, key="login_back"):
                st.session_state.auth_page = "welcome"
                st.rerun()
        with c2:
            if st.button("Create Account", use_container_width=True, key="login_create"):
                st.session_state.auth_page = "register"
                st.rerun()
# ══════════════════════════════════════════════════════════
# REGISTER PAGE
# ══════════════════════════════════════════════════════════

def show_register():
    st.markdown(AUTH_CSS, unsafe_allow_html=True)
    st.markdown('<div class="auth-bg"></div>', unsafe_allow_html=True)

    st.markdown("""
    <style>
    [data-testid="stSidebar"] { display: none !important; }
    .block-container { padding: 0 !important; max-width: 100% !important; }
    section.main > div { padding: 0 !important; }
    </style>
    """, unsafe_allow_html=True)


    st.markdown("""
    <div style="
        display:flex;flex-direction:column;align-items:center;
        padding-top: 12vh;
        margin-bottom: 10px;
    ">
        <div style="
            width:52px;height:52px;border-radius:14px;
            background:linear-gradient(135deg,#4f46e5,#7c3aed);
            display:flex;align-items:center;justify-content:center;
            font-size:24px;
            box-shadow:0 10px 28px rgba(99,102,241,0.32);
            margin-bottom:14px;
        ">🧠</div>
        <div style="
            font-family:'Plus Jakarta Sans',sans-serif;
            font-size:26px;font-weight:800;color:#111827;
            letter-spacing:-0.7px;margin-bottom:4px;text-align:center;
        ">Create account</div>
        <div style="font-size:13px;color:#9ca3af;margin-bottom:24px;text-align:center;">
            Join LexiLens AI to save predictions and reports
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_m, col_r = st.columns([1, 2, 1])
    with col_m:
        with st.form("register_form"):
            username  = st.text_input("Username",         placeholder="Choose a username")
            password  = st.text_input("Password",         type="password", placeholder="Create a password")
            password2 = st.text_input("Confirm Password", type="password", placeholder="Repeat your password")
            submit    = st.form_submit_button("Create Account →", type="primary", use_container_width=True)

        if submit:
            if not username or not password or not password2:
                st.warning("Please fill in all fields.")
            elif len(username) < 3:
                st.error("Username must be at least 3 characters.")
            elif len(password) < 6:
                st.error("Password must be at least 6 characters.")
            elif password != password2:
                st.error("Passwords do not match.")
            else:
                if register_user(username, password):
                    st.session_state.logged_in  = True
                    st.session_state.username   = username
                    st.session_state.history = []
                    st.session_state.auth_page  = "app"
                    st.success(f"Account created! Welcome, {username}!")
                    st.rerun()
                else:
                    st.error("Username already taken. Please choose another.")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("← Back", use_container_width=True):
                st.session_state.auth_page = "welcome"
                st.rerun()
        with c2:
            if st.button("Sign In Instead", use_container_width=True):
                st.session_state.auth_page = "login"
                st.rerun()


# ── AUTH ROUTER ──────────────────────────────────────────
if not st.session_state.logged_in:
    if st.session_state.auth_page == "welcome":
        show_welcome()
    elif st.session_state.auth_page == "login":
        show_login()
    elif st.session_state.auth_page == "register":
        show_register()
    st.stop()

# ── SIDEBAR — sadece login sonrası ───────────────────────
with st.sidebar:
    if not st.session_state.get("logged_in", False):
        st.stop()
    st.markdown(f"""
    <div style="padding:10px 0 16px;">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:16px;">
            <div style="width:40px;height:40px;border-radius:12px;
                        background:linear-gradient(135deg,#4f46e5,#7c3aed);
                        display:flex;align-items:center;justify-content:center;
                        font-size:20px;flex-shrink:0;">🧠</div>
            <div>
                <div style="font-size:18px;font-weight:800;color:white;letter-spacing:-0.5px;">LexiLens AI</div>
                <div style="font-size:9px;color:rgba(255,255,255,0.45);letter-spacing:2px;text-transform:uppercase;margin-top:2px;">v6.0 · Pattern Analysis</div>
            </div>
        </div>
        <div class="user-pill">
            <div class="user-avatar">{st.session_state.username[0].upper()}</div>
            <div>
                <div class="user-name">{st.session_state.username}</div>
                <div style="font-size:10px;color:rgba(255,255,255,0.5);">Signed in</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    page = st.radio("Navigation", [
        "🏠 Dashboard",
        "🔍 Analyse",
        "🎯 Guided Test",
        "📂 Batch Analysis",
        "🤖 AI Assistant",
        "📊 Model Insights",
        "📖 About",
    ], label_visibility="collapsed")

    st.divider()

    st.markdown('<div class="ey" style="margin-bottom:6px;">Age Group</div>', unsafe_allow_html=True)
    age_options = ["Child (5-12)","Teen (13-17)","Adult (18+)"]
    cur = st.session_state.get("age_group","Child (5-12)")
    age_group = st.selectbox("Age", age_options,
                              index=age_options.index(cur) if cur in age_options else 0,
                              label_visibility="collapsed")
    st.session_state.age_group = age_group

    age_info = {
        "Child (5-12)": ("🟡", "Reversals normal up to age 7. Monitor if older."),
        "Teen (13-17)": ("🟠", "Persistent reversals may need assessment."),
        "Adult (18+)":  ("🔴", "May indicate undiagnosed dyslexia."),
    }
    icon, msg = age_info[age_group]
    st.markdown(f"""
    <div style="background:rgba(255,255,255,0.08);border:1px solid rgba(255,255,255,0.15);
                border-radius:10px;padding:10px 12px;margin-top:8px;">
        <div style="font-size:11px;font-weight:700;color:#fff;margin-bottom:4px;">{icon} {age_group}</div>
        <div style="font-size:10px;color:rgba(255,255,255,0.6);line-height:1.5;">{msg}</div>
    </div>
    <div style="font-size:10px;color:rgba(255,255,255,0.35);margin-top:8px;">
        Adjusts result interpretation for {age_group}
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    h = st.session_state.history
    st.markdown('<div class="ey">Recent</div>', unsafe_allow_html=True)
    if not h:
        st.markdown('<div style="color:rgba(255,255,255,0.4);font-size:11px;padding:4px 0;">No predictions yet.</div>', unsafe_allow_html=True)
    else:
        for item in reversed(h[-5:]):
            color = LABEL_COLORS[item["label"]]
            st.markdown(f"""
            <div class="hi">
                <div>
                    <div style="color:{color};font-weight:600;font-size:12px;">{item['label']}</div>
                    <div style="color:rgba(255,255,255,0.4);font-size:10px;">{item['source']} · {item['time']}</div>
                </div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:11px;color:rgba(255,255,255,0.6);">{item['confidence']*100:.0f}%</div>
            </div>""", unsafe_allow_html=True)

        if st.button("🗑 Clear", use_container_width=True):
            st.session_state.history = []
            st.session_state.last_result = None
            st.rerun()

    st.divider()

    st.download_button(
        "⬇ Download PDF",
        generate_pdf(),
        f"lexilens_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
        "application/pdf",
        use_container_width=True,
        key="pdf_sidebar"
    )

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    email_input = st.text_input(
        "Email",
        placeholder="Send report to email...",
        label_visibility="collapsed"
    )
    if st.button("📧 Send Report", use_container_width=True):
        if email_input and "@" in email_input:
            with st.spinner("Sending..."):
                ok, msg = send_pdf_email(
                    email_input,
                    generate_pdf(),
                    st.session_state.username
                )
            if ok:
                st.success("✅ Sent!")
            else:
                st.error(f"❌ {msg}")
        else:
            st.warning("Enter a valid email.")

    st.divider()

    if st.button("Sign Out", use_container_width=True):
        for k in DEFAULTS:
            st.session_state[k] = DEFAULTS[k]
        st.session_state.auth_page = "welcome"
        st.rerun()


# ══════════════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════════════
if st.session_state.get("_from_ai", False):
    st.session_state["_from_ai"] = False
else:
    st.session_state.current_page = page

active_page = st.session_state.current_page

if active_page == "🏠 Dashboard":

    # 1. HERO — tam genişlik
    st.markdown("""
    <div class="card" style="padding:32px;background:linear-gradient(180deg,#4338ca 0%, #312e81 100%);color:white;overflow:hidden;position:relative;margin-bottom:4px;">
    <div style="position:absolute;right:-60px;top:-60px;width:220px;height:220px;border-radius:50%;background:rgba(255,255,255,0.08);"></div>
    <div style="position:absolute;left:-40px;bottom:-40px;width:160px;height:160px;border-radius:50%;background:rgba(255,255,255,0.05);"></div>
    <div style="position:relative;z-index:2">
    <div style="font-size:13px;font-weight:600;opacity:0.7;margin-bottom:8px;letter-spacing:1px;">AI-Powered Educational Screening Platform</div>
    <div style="font-size:40px;font-weight:800;line-height:1;letter-spacing:-2px;margin-bottom:12px;">LexiLens AI</div>
    <div style="font-size:14px;line-height:1.7;opacity:0.85;max-width:600px;">An intelligent handwriting pattern analysis platform designed to support educational dyslexia screening using machine learning and AI-assisted interpretation.</div>
    </div>
    </div>
    """, unsafe_allow_html=True)

    # 2. ETHICAL NOTICE — tam genişlik, ince
    st.markdown("""
    <div style="background:rgba(99,102,241,0.06);border:1px solid rgba(99,102,241,0.15);
                border-left:4px solid #6366f1;border-radius:10px;
                padding:12px 18px;margin-bottom:16px;
                display:flex;align-items:center;gap:12px;">
        <div style="font-size:20px;">⚠️</div>
        <div>
            <span style="font-size:13px;font-weight:700;color:#4f46e5;">Ethical Notice — </span>
            <span style="font-size:13px;color:#64748b;">For educational and research use only. Not a clinical diagnostic tool.</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 3. STAT KARTLARI — 4 kolon
    h = st.session_state.history
    total  = len(h)
    n_norm = sum(1 for x in h if x["label"]=="Normal")
    n_rev  = sum(1 for x in h if x["label"]=="Reversal")
    n_cor  = sum(1 for x in h if x["label"]=="Corrected")
    avg_c  = np.mean([x["confidence"] for x in h])*100 if h else 0

    c1,c2,c3,c4 = st.columns(4)
    for col,val,lbl,clr in [
        (c1, total, "Total Analyses", "#1a1714"),
        (c2, n_rev, "Reversals Found", "#d97706"),
        (c3, f"{avg_c:.0f}%", "Avg Confidence", "#6366f1"),
        (c4, f"{n_rev/total*100:.0f}%" if total else "0%", "Reversal Rate", "#7c3aed"),
    ]:
        col.markdown(f'<div class="sc"><div class="sn" style="color:{clr};">{val}</div><div class="sl">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # 4. ORTA KISIM — sol: How It Works, sağ: Grafik
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown('<div class="ey">How It Works</div>', unsafe_allow_html=True)
        for icon, title, desc in [
            ("📤", "Upload Writing", "Upload handwriting samples in PNG or JPG format for AI analysis."),
            ("🧠", "AI Analysis", "Machine learning models detect handwriting irregularities and patterns."),
            ("📊", "Review Results", "Explore confidence scores, explanations, and educational insights."),
        ]:
            st.markdown(f"""
            <div class="card" style="padding:16px;margin-bottom:8px;display:flex;align-items:flex-start;gap:14px;">
                <div style="font-size:26px;flex-shrink:0;">{icon}</div>
                <div>
                    <div style="font-size:14px;font-weight:700;margin-bottom:4px;">{title}</div>
                    <div style="font-size:12px;color:#94a3b8;line-height:1.6;">{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_right:
        if total >= 3:
            st.markdown('<div class="ey">Prediction Timeline</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(6,2.8))
            fig.patch.set_facecolor('#ffffff')
            ax.set_facecolor('#f9f7f3')
            cmap = {"Normal":"#6366f1","Reversal":"#d97706","Corrected":"#7c3aed"}
            for i,item in enumerate(h):
                ax.scatter(i,item["confidence"]*100,color=cmap[item["label"]],s=60,zorder=3)
            ax.plot(range(len(h)),[x["confidence"]*100 for x in h],color='#4f46e5',linewidth=1,alpha=0.3)
            ax.axhline(70,color='#e5e7eb',linestyle='--',linewidth=1)
            ax.set_ylim(0,105)
            ax.set_xlabel("Prediction #",color='#6b6560',fontsize=9)
            ax.set_ylabel("Confidence %",color='#6b6560',fontsize=9)
            ax.tick_params(colors='#6b6560',labelsize=8)
            for sp in ax.spines.values(): sp.set_color('#e5e7eb')
            ax.grid(True,alpha=0.3,color='#e5e7eb')
            from matplotlib.patches import Patch
            ax.legend(handles=[Patch(facecolor=v,label=k) for k,v in cmap.items()],
                      framealpha=0.8,labelcolor='#1a1714',fontsize=8)
            plt.tight_layout(); st.pyplot(fig); plt.close()

            st.markdown('<div class="ey" style="margin-top:8px;">Distribution</div>', unsafe_allow_html=True)
            sizes = [n_norm,n_rev,n_cor]
            nz = [(s,c,l) for s,c,l in zip(sizes,["#6366f1","#d97706","#7c3aed"],["Normal","Reversal","Corrected"]) if s>0]
            if nz:
                ss,cs,ls = zip(*nz)
                fig2,ax2 = plt.subplots(figsize=(4,2.8))
                fig2.patch.set_facecolor('#ffffff'); ax2.set_facecolor('#ffffff')
                ws,ts,ats = ax2.pie(ss,colors=cs,labels=ls,autopct='%1.0f%%',startangle=90,
                                    pctdistance=0.75,
                                    wedgeprops=dict(width=0.55,edgecolor='#ffffff',linewidth=3))
                for t in ts: t.set_color('#6b6560'); t.set_fontsize(8)
                for at in ats: at.set_color('#1a1714'); at.set_fontsize(8); at.set_fontweight('bold')
                plt.tight_layout(); st.pyplot(fig2); plt.close()
        else:
            st.markdown('<div class="ey">Statistics</div>', unsafe_allow_html=True)
            st.markdown('<div class="emp"><span class="ic">📊</span>Run at least 3 analyses to see statistics.</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # 5. PLATFORM FEATURES — 2 kolon
    st.markdown('<div class="ey">Platform Features</div>', unsafe_allow_html=True)
    f1, f2 = st.columns(2)
    with f1:
        st.markdown("""
        <div class="card" style="margin-bottom:8px;">
            <div style="font-size:15px;font-weight:700;margin-bottom:8px;">🎯 Guided Screening Test</div>
            <div style="color:#64748b;line-height:1.7;font-size:13px;">Step-by-step guided handwriting screening workflow for structured analysis.</div>
        </div>
        <div class="card">
            <div style="font-size:15px;font-weight:700;margin-bottom:8px;">🤖 AI Assistant</div>
            <div style="color:#64748b;line-height:1.7;font-size:13px;">Interactive assistant for dyslexia-related educational guidance and interpretation support.</div>
        </div>
        """, unsafe_allow_html=True)
    with f2:
        st.markdown("""
        <div class="card" style="margin-bottom:8px;">
            <div style="font-size:15px;font-weight:700;margin-bottom:8px;">📂 Batch Processing</div>
            <div style="color:#64748b;line-height:1.7;font-size:13px;">Analyse multiple handwriting samples simultaneously using AI-powered classification.</div>
        </div>
        <div class="card">
            <div style="font-size:15px;font-weight:700;margin-bottom:8px;">📈 Model Insights</div>
            <div style="color:#64748b;line-height:1.7;font-size:13px;">View confusion matrices, calibration curves, ROC analysis, and performance metrics.</div>
        </div>
        """, unsafe_allow_html=True)

    # 6. GEÇMİŞ — tam genişlik

    if h:
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.markdown('<div class="ey">Prediction History</div>', unsafe_allow_html=True)
        import pandas as pd
        tab1, tab2 = st.tabs(["This Session", "All History"])
        with tab1:
            df = pd.DataFrame([{"Time":x["time"],"Source":x["source"],
                                "Prediction":x["label"],"Confidence":f"{x['confidence']*100:.1f}%"}
                               for x in reversed(h[-10:])])
            st.dataframe(df, use_container_width=True, hide_index=True)
        with tab2:
            try:
                conn = sqlite3.connect("lexilens.db")
                df_all = pd.read_sql_query(
                    "SELECT timestamp,label,ROUND(confidence*100,1)||'%' as confidence,source,age_group FROM predictions WHERE username=? ORDER BY id DESC LIMIT 100",
                    conn, params=(st.session_state.username,)
                )
                conn.close()
                if len(df_all)>0:
                    st.dataframe(df_all, use_container_width=True, hide_index=True)
                else:
                    st.markdown('<div style="color:var(--t3);font-size:12px;">No stored predictions yet.</div>', unsafe_allow_html=True)
            except:
                st.markdown('<div style="color:var(--t3);font-size:12px;">Database unavailable.</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# ANALYSE
# ══════════════════════════════════════════════════════════
elif active_page == "🔍 Analyse":
    col_in, col_out = st.columns([1.3,1], gap="large")

    with col_in:
        st.markdown('<div class="ey">Input</div>', unsafe_allow_html=True)
        up = st.file_uploader("Upload", type=["png","jpg","jpeg"], label_visibility="collapsed")
        if up:
            img_pil = Image.open(up)
            st.image(img_pil.resize((200,200)), caption="Uploaded — resized to 28x28 for analysis")
            img_array = preprocess(img_pil)
            label,conf,probs = predict(img_array)
            add_history(label,conf,"Upload")
            st.session_state.last_result = (label,conf,probs)

    with col_out:
        st.markdown('<div class="ey">Analysis Results</div>', unsafe_allow_html=True)
        if st.session_state.last_result:
            st.markdown(f"""
            <div style="background:rgba(99,102,241,0.06);border:1px solid rgba(99,102,241,0.15);
                        border-radius:10px;padding:10px 14px;margin-bottom:12px;
                        display:flex;align-items:center;gap:10px;">
                <div style="font-size:18px;">👤</div>
                <div>
                    <div style="font-size:11px;font-weight:700;color:#4f46e5;">Interpreting for: {st.session_state.age_group}</div>
                    <div style="font-size:11px;color:#6b7280;margin-top:2px;">Results are contextualised for this age group</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            render_result(*st.session_state.last_result)
        else:
            st.markdown('<div class="emp"><span class="ic">✍️</span>Upload a handwritten letter to begin analysis.</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# GUIDED TEST
# ══════════════════════════════════════════════════════════
elif active_page == "🎯 Guided Test":
    st.markdown('<div class="ey">Guided Screening Test</div>', unsafe_allow_html=True)

    if st.session_state.test_phase == "idle":
        st.markdown("""
        <div class="card" style="max-width:640px;">
            <div class="ey">How the test works</div>
            <div style='font-size:13px;color:var(--t2);line-height:1.9;'>
                This test asks you to write <b style='color:var(--t1);'>8 specific characters</b>
                one at a time — the characters most commonly associated with reversal patterns in dyslexia.
                Write each naturally on paper, photograph it, and upload it below.
                <span style="color:var(--gr);">✦</span> Write each character as you normally would — no hints are given
                <span style="color:var(--am);">✦</span> You can skip any character if needed
                <span style="color:var(--pu);">✦</span> A risk-level summary report is produced at the end
                <span style='color:var(--t3);font-size:11px;'>For educational purposes only. Not a clinical diagnosis.</span>
            </div>
        </div>
        <div class="card" style="max-width:640px;">
            <div class="ey">Characters You Will Write</div>
            <div style='display:flex;gap:10px;flex-wrap:nowrap;'>
        """ + "".join([
            f'<div style="width:52px;height:52px;border-radius:12px;background:var(--hover);'
            f'border:1.5px solid var(--border);display:flex;align-items:center;justify-content:center;'
            f'font-family:Fraunces,serif;font-size:24px;font-weight:700;color:var(--t1);">'
            f'{c["char"]}</div>'
            for c in TEST_SEQ
        ]) + """
            </div>
            <div style='font-size:11px;color:var(--t3);margin-top:12px;'>
                No hints provided — write naturally to ensure authentic pattern detection.
            </div>
        </div>
        """, unsafe_allow_html=True)

        col_btn, _ = st.columns([1,3])
        if col_btn.button("▶ Start Test", type="primary", use_container_width=True):
            st.session_state.test_phase   = "running"
            st.session_state.test_step    = 0
            st.session_state.test_results = []
            st.rerun()

    elif st.session_state.test_phase == "running":
        step  = st.session_state.test_step
        total = len(TEST_SEQ)
        info  = TEST_SEQ[step]
        timer_key = f"step_start_{step}"
        if timer_key not in st.session_state:
            st.session_state[timer_key] = time.time()

        pct = int(step/total*100)
        dots_html = '<div class="dots">' + "".join([
            f'<div class="dot {"dot-active" if i==step else "dot-done" if i<step else ""}">'
            f'{TEST_SEQ[i]["char"]}</div>'
            for i in range(total)
        ]) + '</div>'

        st.markdown(f"""
        <div style='margin-bottom:22px;'>
            <div style='display:flex;justify-content:space-between;margin-bottom:6px;'>
                <div class="ey" style="margin-bottom:0;">Character {step+1} of {total}</div>
                <div style='font-family:"JetBrains Mono",monospace;font-size:10px;color:var(--t3);'>{pct}% complete</div>
            </div>
            <div style='height:6px;background:rgba(0,0,0,0.06);border-radius:99px;overflow:hidden;margin-bottom:10px;'>
                <div style='height:100%;width:{pct}%;background:linear-gradient(90deg,#4f46e5,#7c3aed);
                            border-radius:99px;transition:width .4s;'></div>
            </div>
            {dots_html}
        </div>
        """, unsafe_allow_html=True)

        col_left, col_right = st.columns([1.2,1], gap="large")

        with col_left:
            st.markdown(f"""
            <div class="tp">
                <div style='font-family:"JetBrains Mono",monospace;font-size:9px;letter-spacing:3px;
                            text-transform:uppercase;color:rgba(255,255,255,0.7);'>Please write</div>
                <div class="tchar">{info["char"]}</div>
                <div class="thint">{info["hint"]}</div>
            </div>
            """, unsafe_allow_html=True)

            uploaded_test = st.file_uploader("Upload photo",type=["png","jpg","jpeg"],
                                              label_visibility="collapsed",key=f"tu_{step}")
            img_array = None
            if uploaded_test is not None:
                img_pil = Image.open(uploaded_test)
                st.image(img_pil.resize((180,180)), caption="Uploaded")
                img_array = preprocess(img_pil)

            cs, ck = st.columns(2)
            submit = cs.button("Analyse", type="primary", use_container_width=True, key=f"sub_{step}")
            skip   = ck.button("Skip", use_container_width=True, key=f"skp_{step}")

            if submit:
                if img_array is not None:
                    label,conf,probs = predict(img_array)
                    rt = round(time.time()-st.session_state.get(timer_key,time.time()),1)
                    st.session_state.test_results.append({
                        "char":info["char"],"label":label,"confidence":conf,
                        "probs":probs,"skipped":False,"response_time":rt,
                    })
                    add_history(label,conf,f"Test·{info['char']}")
                    save_test_result(info["char"],label,conf,rt,
                                     st.session_state.age_group,
                                     st.session_state.username)
                    if step+1>=total: st.session_state.test_phase="done"
                    else: st.session_state.test_step=step+1
                    st.rerun()
                else:
                    st.warning("Please upload a photo first.")

            if skip:
                st.session_state.test_results.append({
                    "char":info["char"],"label":"Skipped","confidence":0,
                    "probs":{},"skipped":True,"response_time":0,
                })
                if step+1>=total: st.session_state.test_phase="done"
                else: st.session_state.test_step=step+1
                st.rerun()

        with col_right:
            st.markdown('<div class="ey">Results So Far</div>', unsafe_allow_html=True)
            if not st.session_state.test_results:
                st.markdown('<div class="emp" style="padding:24px;"><span class="ic" style="font-size:18px;">📸</span>Upload a photo and click Analyse.</div>', unsafe_allow_html=True)
            else:
                for r in reversed(st.session_state.test_results):
                    if r["skipped"]:
                        st.markdown(f'<div class="hi"><div style="font-family:Fraunces,serif;font-size:20px;font-weight:700;color:var(--t3);">{r["char"]}</div><div style="color:var(--t3);font-size:11px;">Skipped</div></div>', unsafe_allow_html=True)
                    else:
                        color = LABEL_COLORS.get(r["label"],"#888")
                        icon  = "✅" if r["label"]=="Normal" else "⚠️" if r["label"]=="Reversal" else "🔄"
                        bw    = int(r["confidence"]*100)
                        bf    = LABEL_BAR.get(r["label"],"bfn")
                        rt    = r.get("response_time",0)
                        rt_c  = "#dc2626" if rt>10 else "#d97706" if rt>5 else "#6366f1"
                        st.markdown(f"""
                        <div class="hi" style="padding:12px 14px;">
                            <div style='font-family:Fraunces,serif;font-size:18px;font-weight:700;color:var(--t1);width:36px;'>{r['char']}</div>
                            <div style='flex:1;margin:0 12px;'>
                                <div style='display:flex;justify-content:space-between;margin-bottom:4px;'>
                                    <span style='color:{color};font-weight:600;font-size:12px;'>{icon} {r['label']}</span>
                                    <span style='font-family:"JetBrains Mono",monospace;font-size:11px;color:var(--t2);'>{r['confidence']*100:.0f}%</span>
                                </div>
                                <div class="bt"><div class="{bf}" style="width:{bw}%;"></div></div>
                            </div>
                            <div style='text-align:right;'>
                                <div style='font-family:"JetBrains Mono",monospace;font-size:12px;color:{rt_c};font-weight:600;'>{rt}s</div>
                            </div>
                        </div>""", unsafe_allow_html=True)

        if st.button("✕ Cancel Test"):
            st.session_state.test_phase="idle"
            st.session_state.test_step=0
            st.session_state.test_results=[]
            st.rerun()

    elif st.session_state.test_phase == "done":
        results   = st.session_state.test_results
        completed = [r for r in results if not r["skipped"]]
        n_rev  = sum(1 for r in completed if r["label"]=="Reversal")
        n_norm = sum(1 for r in completed if r["label"]=="Normal")
        total  = len(completed)
        rate   = n_rev/total*100 if total else 0

        if rate==0: risk,rc,rd = "Low Risk","#6366f1","No reversal patterns detected."
        elif rate<=37.5: risk,rc,rd = "Moderate","#d97706","Some reversals detected. Consider monitoring."
        else: risk,rc,rd = "Elevated","#dc2626","Multiple reversals detected. Professional assessment recommended."

        st.markdown(f"""
        <div class="rc" style="max-width:700px;margin:0 auto 28px;">
            <div style='font-family:"JetBrains Mono",monospace;font-size:9px;letter-spacing:3px;
                        text-transform:uppercase;color:rgba(255,255,255,0.7);margin-bottom:12px;'>Screening Complete</div>
            <div style='font-family:Fraunces,serif;font-size:44px;font-weight:700;
                        color:#fff;letter-spacing:-1px;margin-bottom:8px;'>{risk}</div>
            <div style='font-size:13px;color:rgba(255,255,255,0.8);max-width:420px;margin:0 auto;line-height:1.6;'>{rd}</div>
        </div>""", unsafe_allow_html=True)

        c1,c2,c3,c4 = st.columns(4)
        for col,val,lbl,clr in [(c1,total,"Tested","#1a1714"),(c2,n_rev,"Reversals","#dc2626"),(c3,n_norm,"Normal","#6366f1"),(c4,f"{rate:.0f}%","Reversal Rate",rc)]:
            col.markdown(f'<div class="sc"><div class="sn" style="font-size:30px;color:{clr};">{val}</div><div class="sl">{lbl}</div></div>', unsafe_allow_html=True)

        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        st.markdown('<div class="ey">Detailed Results</div>', unsafe_allow_html=True)

        for r in results:
            if r["skipped"]:
                st.markdown(f'<div class="hi"><div style="font-family:Fraunces,serif;font-size:20px;font-weight:700;color:var(--t3);width:36px;">{r["char"]}</div><div style="flex:1;color:var(--t3);font-size:12px;margin-left:12px;">Skipped</div></div>', unsafe_allow_html=True)
            else:
                color=LABEL_COLORS.get(r["label"],"#888")
                icon="✅" if r["label"]=="Normal" else "⚠️" if r["label"]=="Reversal" else "🔄"
                bw=int(r["confidence"]*100)
                bf=LABEL_BAR.get(r["label"],"bfn")
                rt=r.get("response_time",0)
                rt_c="#dc2626" if rt>10 else "#d97706" if rt>5 else "#6366f1"
                st.markdown(f"""
                <div class="hi" style="padding:14px 16px;">
                    <div style='font-family:Fraunces,serif;font-size:18px;font-weight:700;color:var(--t1);width:40px;'>{r['char']}</div>
                    <div style='flex:1;margin:0 16px;'>
                        <div style='display:flex;justify-content:space-between;margin-bottom:5px;'>
                            <span style='color:{color};font-weight:600;font-size:12px;'>{icon} {r['label']}</span>
                            <span style='font-family:"JetBrains Mono",monospace;font-size:11px;color:var(--t2);'>{r['confidence']*100:.0f}%</span>
                        </div>
                        <div class="bt"><div class="{bf}" style="width:{bw}%;"></div></div>
                    </div>
                    <div style='text-align:right;min-width:60px;'>
                        <div style='font-family:"JetBrains Mono",monospace;font-size:12px;color:{rt_c};font-weight:600;'>{rt}s</div>
                        <div style='font-size:9px;color:var(--t3);'>response</div>
                    </div>
                </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div style='background:rgba(220,38,38,0.04);border:1px solid rgba(220,38,38,0.12);
                    border-left:3px solid #dc2626;border-radius:0 10px 10px 0;
                    padding:14px 18px;margin-top:16px;font-size:12px;color:#7f1d1d;line-height:1.7;'>
            <b>Important:</b> This screening test is for educational purposes only.
            Results should not be interpreted as a clinical diagnosis.
            Consult a qualified educational psychologist if you have concerns.
        </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        c1,c2,c3,_ = st.columns([1,1,1,1])
        if c1.button("Retake", type="primary", use_container_width=True):
            st.session_state.test_phase="running"
            st.session_state.test_step=0
            st.session_state.test_results=[]
            st.rerun()
        if c2.button("🤖 Ask AI", use_container_width=True):
            rev_chars = [r["char"] for r in completed if r["label"] == "Reversal"]
            ctx = (f"Guided test completed. {n_rev}/{total} reversals detected. "
                f"Risk level: {risk}. Reversed characters: {rev_chars}. "
                f"Age group: {st.session_state.age_group}.")
            st.session_state.chat_ctx = ctx
            st.session_state.chat_messages = [{
                "role": "assistant",
                "content": (
                    f"👋 I can see you just completed the guided screening test.\n\n"
                    f"**Results:** {n_rev} out of {total} characters showed reversal patterns — "
                    f"Risk level: **{risk}**.\n\n"
                    f"Would you like me to explain what this means, or do you have specific questions?"
                )
            }]
            st.session_state.current_page = "🤖 AI Assistant"
            st.rerun()


# ══════════════════════════════════════════════════════════
# BATCH
# ══════════════════════════════════════════════════════════
elif active_page == "📂 Batch Analysis":
    st.markdown('<div class="ey">Batch Processing</div>', unsafe_allow_html=True)
    files = st.file_uploader("Upload",type=["png","jpg","jpeg"],accept_multiple_files=True,label_visibility="collapsed")
    if files:
        import pandas as pd
        results=[]
        cols=st.columns(min(len(files),4))
        for i,f in enumerate(files):
            img_pil=Image.open(f)
            flat=preprocess(img_pil)
            label,conf,probs=predict(flat)
            results.append({"File":f.name,"Prediction":label,"Confidence":f"{conf*100:.1f}%"})
            add_history(label,conf,"Batch")
            with cols[i%4]:
                st.image(img_pil.resize((100,100)),caption=f.name[:14])
                color=LABEL_COLORS[label]
                st.markdown(f'<div style="text-align:center;"><span style="color:{color};font-weight:700;font-size:11px;">{label}</span></div>', unsafe_allow_html=True)
        st.markdown("---")
        df=pd.DataFrame(results)
        st.dataframe(df,use_container_width=True,hide_index=True)
        total=len(results)
        n_rev=sum(1 for r in results if r["Prediction"]=="Reversal")
        c1,c2,c3=st.columns(3)
        for col,val,lbl,clr in [(c1,total,"Total","#1a1714"),(c2,n_rev,"Reversals","#d97706"),(c3,f"{n_rev/total*100:.1f}%","Rate","#7c3aed")]:
            col.markdown(f'<div class="sc"><div class="sn" style="font-size:28px;color:{clr};">{val}</div><div class="sl">{lbl}</div></div>', unsafe_allow_html=True)
        st.download_button("⬇ Download CSV",df.to_csv(index=False).encode(),"results.csv","text/csv")


# ══════════════════════════════════════════════════════════
# AI ASSISTANT
# ══════════════════════════════════════════════════════════
elif active_page == "🤖 AI Assistant":

    if len(st.session_state.chat_messages) == 0:
        st.session_state.chat_messages = [
            {
                "role": "assistant",
                "content": (
                    "Hi! 👋 I'm the LexiLens Assistant.\n\n"
                    "I can help explain dyslexia signs, handwriting reversals, "
                    "analysis results, and what different patterns may mean.\n\n"
                    "You can ask me things like:\n"
                    "• What does letter reversal mean?\n"
                    "• Is this a dyslexia sign?\n"
                    "• Should I seek professional support?\n"
                    "• How accurate is the analysis?\n\n"
                    "What would you like to explore first? ✨"
                )
            }
        ]

    st.markdown('<div class="ey">LexiLens Dyslexia Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:13px;color:var(--t2);margin-bottom:20px;line-height:1.6;">Ask me anything about dyslexia, handwriting patterns, or your results.</div>', unsafe_allow_html=True)

    sugs=["What does a Reversal mean?","Should I see a specialist?","What is dyslexia?","How accurate is this tool?"]

    # Guided test sonucu varsa butona ekle
    if st.session_state.get("chat_ctx"):
        sugs = ["📊 Analyse my results"] + sugs[:3]

    cols=st.columns(4)
    for i,(col,s) in enumerate(zip(cols,sugs)):
        with col:
            if st.button(s,key=f"s{i}",use_container_width=True):
                if s == "📊 Analyse my results":
                    user_msg = f"Please analyse my guided test results. Context: {st.session_state.chat_ctx}"
                else:
                    user_msg = s
                st.session_state.chat_messages.append({"role":"user","content":user_msg})
                with st.spinner("Thinking..."):
                    resp=get_ai_response(user_msg)
                st.session_state.chat_messages.append({"role":"assistant","content":resp})
                st.rerun()

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    for msg in st.session_state.chat_messages:
        if msg["role"]=="user":
            st.markdown(f'<div class="cu"><div class="cl clu">YOU</div>{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="cb"><div class="cl clb">LEXILENS ASSISTANT</div>{msg["content"]}</div>', unsafe_allow_html=True)

    user_input=st.chat_input("Ask about dyslexia or your results...")
    if user_input:
        st.session_state.chat_started = True
        st.session_state.chat_messages.append({"role":"user","content":user_input})
        with st.spinner("Thinking..."):
            resp=get_ai_response(user_input)
        st.session_state.chat_messages.append({"role":"assistant","content":resp})
        st.rerun()

    if st.session_state.chat_messages:
        if st.button("🗑 Clear Chat"):
            st.session_state.chat_messages=[]
            st.session_state.chat_ctx=""
            st.rerun()


# ══════════════════════════════════════════════════════════
# MODEL INSIGHTS
# ══════════════════════════════════════════════════════════
elif active_page == "📊 Model Insights":
    st.markdown('<div class="ey">Model Performance & Insights</div>', unsafe_allow_html=True)

    c1,c2,c3,c4=st.columns(4)
    for col,val,lbl,clr in [
        (c1,"85.85%","Test Accuracy","#6366f1"),
        (c2,"85.95%","Macro F1","#4f46e5"),
        (c3,"151,649","Training Images","#7c3aed"),
        (c4,"91.9/100","SUS Usability","#d97706"),
    ]:
        col.markdown(f'<div class="sc"><div class="sn" style="font-size:26px;color:{clr};">{val}</div><div class="sl">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="ey">Model Comparison</div>', unsafe_allow_html=True)
    import pandas as pd
    df=pd.DataFrame([
        {"Model":"Random Forest (Deployed)","Test Acc":"85.85%","Macro F1":"85.95%","CV Acc":"87.07%","Status":"Active"},
        {"Model":"SVM RBF","Test Acc":"N/A","Macro F1":"N/A","CV Acc":"90.01%","Status":"CV Only"},
        {"Model":"CNN","Test Acc":"76.45%","Macro F1":"76.35%","CV Acc":"N/A","Status":"Evaluated"},
        {"Model":"Gradient Boosting","Test Acc":"75.90%","Macro F1":"75.78%","CV Acc":"83.47%","Status":"Evaluated"},
        {"Model":"Logistic Regression","Test Acc":"70.51%","Macro F1":"70.12%","CV Acc":"76.12%","Status":"Baseline"},
    ])
    st.dataframe(df,use_container_width=True,hide_index=True)

    graphs = {
        "Confusion Matrix":    ("assets/graphs/cm_best_classical.png",    "Shows how well the model classifies each pattern. Diagonal = correct predictions."),
        "ROC / AUC":           ("assets/graphs/roc_auc_classical.png",    "Area under curve measures discrimination ability. Closer to 1.0 = better."),
        "Calibration":         ("assets/graphs/calibration_curve_before.png", "Shows whether confidence scores match actual accuracy."),
        "Feature Importance":  ("assets/graphs/feature_importance.png",   "Pixel regions most influential in the model's decisions."),
        "Learning Curve":      ("assets/graphs/learning_curve.png",       "Shows how accuracy improves with more training data."),
        "CNN Training":        ("assets/graphs/cnn_training_history.png", "Training and validation accuracy across epochs."),
        "Model Comparison":    ("assets/graphs/model_comparison.png",     "Side-by-side accuracy and F1 comparison across all models."),
        "t-SNE / PCA":         ("assets/graphs/viz_pca_tsne.png",         "2D visualisation of how the model separates the three pattern classes."),
    }

    available = {k: v for k, (v, _) in graphs.items() if os.path.exists(v)}
    descriptions = {k: d for k, (_, d) in graphs.items()}

    if available:
        keys = list(available.keys())
        for i in range(0, len(keys), 2):
            cols = st.columns(2)
            for j, col in enumerate(cols):
                if i + j < len(keys):
                    k = keys[i + j]
                    with col:
                        st.markdown(f'<div class="ey">{k}</div>', unsafe_allow_html=True)
                        st.markdown(f'<div style="font-size:11px;color:var(--t3);margin-bottom:6px;">{descriptions[k]}</div>', unsafe_allow_html=True)
                        st.image(available[k], use_container_width=True)
    else:
        st.markdown('<div class="emp"><span class="ic">📊</span>Run the Jupyter notebook to generate visualisations.</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="ey">Calibration Results (ECE)</div>', unsafe_allow_html=True)
    df_cal=pd.DataFrame([
        {"Class":"Normal","Before":"0.0655","After":"0.0259","Improvement":"60.5%"},
        {"Class":"Reversal","Before":"0.0707","After":"0.0209","Improvement":"70.4%"},
        {"Class":"Corrected","Before":"0.0800","After":"0.0075","Improvement":"90.6%"},
    ])
    st.dataframe(df_cal,use_container_width=True,hide_index=True)


# ══════════════════════════════════════════════════════════
# ABOUT
# ══════════════════════════════════════════════════════════
elif active_page == "📖 About":
    col1,col2=st.columns([1,1],gap="large")
    with col1:
        st.markdown("""
        <div class="card">
            <div class="ey">About LexiLens AI</div>
            <div style='font-size:14px;color:var(--t2);line-height:1.8;'>
                <b style='color:var(--t1);'>LexiLens AI</b> is an AI-powered handwriting pattern
                analyser developed as a final year university project. It combines machine learning
                classification with an AI assistant to support dyslexia screening.
            </div>
        </div>
        <div class="card">
            <div class="ey">How It Works</div>
            <div style='font-size:13px;color:var(--t2);line-height:1.9;'>
                <b style='color:var(--t1);'>1. Input</b> — Upload or photograph a character.
                <b style='color:var(--t1);'>2. Preprocess</b> — Greyscale, resize to 28x28 px.
                <b style='color:var(--t1);'>3. Classify</b> — Random Forest model predicts class.
                <b style='color:var(--t1);'>4. Calibrate</b> — Isotonic calibration improves confidence.
                <b style='color:var(--t1);'>5. Report</b> — Prediction, probabilities, explanation.
                <b style='color:var(--t1);'>6. Assist</b> — AI chatbot answers follow-up questions.
            </div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="card">
            <div class="ey">Classification Classes</div>
            <div style='font-size:13px;color:var(--t2);line-height:2;'>
                <span style='color:#6366f1;font-weight:600;'>Normal</span> — Correct orientation.
                <span style='color:#d97706;font-weight:600;'>Reversal</span> — Mirror/rotational reversal.
                <span style='color:#7c3aed;font-weight:600;'>Corrected</span> — Self-corrected during writing.
            </div>
        </div>
        <div class="card">
            <div class="ey">Performance</div>
            <div style='font-size:13px;color:var(--t2);line-height:2;'>
                <b style='color:var(--t1);'>Algorithm</b> — Random Forest
                <b style='color:var(--t1);'>Test Accuracy</b> — <span style='color:#6366f1;font-weight:600;'>85.85%</span>
                <b style='color:var(--t1);'>Training Data</b> — 151,649 images
                <b style='color:var(--t1);'>SUS Score</b> — <span style='color:#6366f1;font-weight:600;'>91.9 / 100</span>
            </div>
        </div>
        <div class="card" style="border-left:3px solid #d97706;">
            <div class="ey">Disclaimer</div>
            <div style='font-size:12px;color:var(--t3);line-height:1.7;'>
                For <b style='color:var(--t2);'>educational and research purposes only</b>.
                Not a clinical diagnostic tool. Consult a qualified professional for formal assessment.
            </div>
        </div>""", unsafe_allow_html=True)