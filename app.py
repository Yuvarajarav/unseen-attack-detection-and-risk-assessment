"""
Unseen Attack Detection System — UNSW-NB15
Uses XGBoost (Binary + Multiclass) with 26 features
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Cyber Attack Detection",
    page_icon="🛡️",
    layout="wide"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Rajdhani', sans-serif;
    }

    .stApp {
        background: #0a0e1a;
        color: #c9d1d9;
    }

    .main-title {
        font-family: 'Share Tech Mono', monospace;
        font-size: 2.2rem;
        color: #00e5ff;
        text-align: center;
        letter-spacing: 3px;
        margin-bottom: 0;
        text-shadow: 0 0 20px #00e5ff55;
    }

    .sub-title {
        text-align: center;
        color: #8b949e;
        font-size: 1rem;
        letter-spacing: 2px;
        margin-top: 4px;
        margin-bottom: 30px;
    }

    .section-header {
        font-family: 'Share Tech Mono', monospace;
        color: #00e5ff;
        font-size: 1rem;
        letter-spacing: 2px;
        border-left: 3px solid #00e5ff;
        padding-left: 10px;
        margin: 20px 0 12px 0;
        text-transform: uppercase;
    }

    .card {
        background: #161b27;
        border: 1px solid #1e2d40;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 16px;
    }

    .result-normal {
        background: linear-gradient(135deg, #0d2818, #0a1f14);
        border: 1px solid #238636;
        border-radius: 10px;
        padding: 24px;
        text-align: center;
    }

    .result-attack {
        background: linear-gradient(135deg, #2d0f0f, #1a0808);
        border: 1px solid #da3633;
        border-radius: 10px;
        padding: 24px;
        text-align: center;
    }

    .result-label {
        font-family: 'Share Tech Mono', monospace;
        font-size: 1.8rem;
        font-weight: bold;
        letter-spacing: 3px;
    }

    .attack-type-badge {
        display: inline-block;
        background: #1c2d3a;
        border: 1px solid #00e5ff44;
        border-radius: 20px;
        padding: 4px 16px;
        font-family: 'Share Tech Mono', monospace;
        color: #00e5ff;
        font-size: 1rem;
        margin-top: 8px;
    }

    .risk-bar-container {
        background: #1e2d40;
        border-radius: 8px;
        height: 16px;
        width: 100%;
        overflow: hidden;
        margin: 10px 0;
    }

    .metric-box {
        background: #161b27;
        border: 1px solid #1e2d40;
        border-radius: 8px;
        padding: 14px;
        text-align: center;
    }

    .metric-value {
        font-family: 'Share Tech Mono', monospace;
        font-size: 1.6rem;
        color: #00e5ff;
        display: block;
    }

    .metric-label {
        font-size: 0.8rem;
        color: #8b949e;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    .tip-box {
        background: #0d1b2a;
        border: 1px solid #1e3a5f;
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 0.85rem;
        color: #8b949e;
        margin-bottom: 10px;
    }

    div[data-testid="stNumberInput"] label,
    div[data-testid="stSelectbox"] label {
        color: #8b949e !important;
        font-size: 0.82rem !important;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    div[data-testid="stNumberInput"] input {
        background: #0d1117 !important;
        border: 1px solid #1e2d40 !important;
        color: #c9d1d9 !important;
        border-radius: 6px !important;
        font-family: 'Share Tech Mono', monospace !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #00e5ff22, #00e5ff11) !important;
        border: 1px solid #00e5ff !important;
        color: #00e5ff !important;
        font-family: 'Share Tech Mono', monospace !important;
        letter-spacing: 2px !important;
        font-size: 1rem !important;
        padding: 12px !important;
        border-radius: 8px !important;
        transition: all 0.3s !important;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #00e5ff44, #00e5ff22) !important;
        box-shadow: 0 0 20px #00e5ff33 !important;
    }

    div[data-testid="stSelectbox"] > div {
        background: #0d1117 !important;
        border: 1px solid #1e2d40 !important;
        color: #c9d1d9 !important;
    }

    hr {
        border-color: #1e2d40 !important;
    }

    .stSuccess, .stError, .stInfo, .stWarning {
        border-radius: 8px !important;
    }

    /* Hide streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# RISK SCORE MAP
# ─────────────────────────────────────────────
RISK_SCORES = {
    'Normal':         0,
    'Reconnaissance': 30,
    'Fuzzers':        40,
    'Analysis':       40,
    'Generic':        50,
    'DoS':            70,
    'Exploits':       80,
    'Backdoor':       90,
    'Backdoors':      90,
    'Shellcode':      95,
    'Worms':          100,
}

def get_risk_level(score):
    if score == 0:    return "SAFE", "#238636"
    elif score <= 30: return "LOW RISK", "#d29922"
    elif score <= 50: return "MEDIUM RISK", "#e3812b"
    elif score <= 75: return "HIGH RISK", "#da3633"
    else:             return "CRITICAL", "#ff0000"

# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    try:
        xgb_binary     = joblib.load('xgb_binary.pkl')
        xgb_multiclass = joblib.load('xgb_multiclass.pkl')
        scaler         = joblib.load('scaler.pkl')
        label_encoders = joblib.load('label_encoders.pkl')
        target_encoder = joblib.load('target_encoder.pkl')
        metadata       = joblib.load('metadata.pkl')
        return xgb_binary, xgb_multiclass, scaler, label_encoders, target_encoder, metadata
    except Exception as e:
        st.error(f"❌ Error loading model files: {e}")
        st.info("Make sure all .pkl files are in the same folder as app.py")
        st.stop()

xgb_binary, xgb_multiclass, scaler, label_encoders, target_encoder, metadata = load_models()
SELECTED_FEATURES = metadata['selected_features']

# ─────────────────────────────────────────────
# PROTO / SERVICE / STATE OPTIONS
# (from LabelEncoder classes saved during training)
# ─────────────────────────────────────────────
proto_classes   = list(label_encoders['proto'].classes_)
service_classes = list(label_encoders['service'].classes_)
state_classes   = list(label_encoders['state'].classes_)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown('<h1 class="main-title">🛡️ CYBER ATTACK DETECTION SYSTEM</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">UNSW-NB15 · XGBoost · Real-Time Threat Analysis</p>', unsafe_allow_html=True)
st.markdown("---")

# Model info bar
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown('<div class="metric-box"><span class="metric-value">XGBoost</span><span class="metric-label">Model</span></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="metric-box"><span class="metric-value">95.04%</span><span class="metric-label">Binary Accuracy</span></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="metric-box"><span class="metric-value">82.31%</span><span class="metric-label">Attack Type Accuracy</span></div>', unsafe_allow_html=True)
with c4:
    st.markdown('<div class="metric-box"><span class="metric-value">10</span><span class="metric-label">Attack Categories</span></div>', unsafe_allow_html=True)

st.markdown("---")

# ─────────────────────────────────────────────
# INPUT FORM
# ─────────────────────────────────────────────
st.markdown('<p class="section-header">// Network Traffic Features</p>', unsafe_allow_html=True)

st.markdown('<div class="tip-box">💡 <b>Tip:</b> All 26 features below are the most important ones selected from the UNSW-NB15 dataset. Fill in the network traffic values and click Analyze.</div>', unsafe_allow_html=True)

with st.form("detection_form"):

    # ── ROW 1: Categorical features ──────────────────
    st.markdown("**Protocol & Connection Info**")
    r1c1, r1c2, r1c3 = st.columns(3)
    with r1c1:
        proto_sel   = st.selectbox("Protocol (proto)", proto_classes, index=proto_classes.index('tcp') if 'tcp' in proto_classes else 0)
    with r1c2:
        service_sel = st.selectbox("Service", service_classes, index=service_classes.index('-') if '-' in service_classes else 0)
    with r1c3:
        state_sel   = st.selectbox("Connection State (state)", state_classes, index=state_classes.index('FIN') if 'FIN' in state_classes else 0)

    st.markdown("**Traffic Volume**")
    r2c1, r2c2, r2c3, r2c4 = st.columns(4)
    with r2c1:
        sbytes = st.number_input("Source Bytes (sbytes)", min_value=0.0, value=258.0, help="Total bytes sent by source")
    with r2c2:
        dbytes = st.number_input("Dest Bytes (dbytes)", min_value=0.0, value=172.0, help="Total bytes sent by destination")
    with r2c3:
        spkts  = st.number_input("Source Packets (spkts)", min_value=0.0, value=6.0, help="Number of packets sent by source")
    with r2c4:
        dpkts  = st.number_input("Dest Packets (dpkts)", min_value=0.0, value=4.0, help="Number of packets sent by destination")

    st.markdown("**Load & Rate**")
    r3c1, r3c2, r3c3, r3c4 = st.columns(4)
    with r3c1:
        rate  = st.number_input("Rate", min_value=0.0, value=74.08, help="Total packets per second")
    with r3c2:
        sload = st.number_input("Source Load (sload)", min_value=0.0, value=14158.94, help="Source bits per second")
    with r3c3:
        dload = st.number_input("Dest Load (dload)", min_value=0.0, value=8495.36, help="Destination bits per second")
    with r3c4:
        dur   = st.number_input("Duration (dur)", min_value=0.0, value=0.121, format="%.6f", help="Record total duration")

    st.markdown("**Timing & Jitter**")
    r4c1, r4c2, r4c3, r4c4 = st.columns(4)
    with r4c1:
        sinpkt = st.number_input("Src Inter-Pkt Time (sinpkt)", min_value=0.0, value=24.29, help="Source interpacket arrival time (ms)")
    with r4c2:
        dinpkt = st.number_input("Dst Inter-Pkt Time (dinpkt)", min_value=0.0, value=8.37, help="Destination interpacket arrival time (ms)")
    with r4c3:
        sjit   = st.number_input("Source Jitter (sjit)", min_value=0.0, value=30.17, help="Source jitter (ms)")
    with r4c4:
        djit   = st.number_input("Dest Jitter (djit)", min_value=0.0, value=11.83, help="Destination jitter (ms)")

    st.markdown("**TCP Handshake**")
    r5c1, r5c2, r5c3, r5c4 = st.columns(4)
    with r5c1:
        tcprtt = st.number_input("TCP RTT (tcprtt)", min_value=0.0, value=0.0, format="%.6f", help="TCP connection setup round-trip time")
    with r5c2:
        synack = st.number_input("SYN-ACK Time (synack)", min_value=0.0, value=0.0, format="%.6f", help="Time between SYN and SYN-ACK")
    with r5c3:
        ackdat = st.number_input("ACK-Data Time (ackdat)", min_value=0.0, value=0.0, format="%.6f", help="Time between SYN-ACK and ACK")
    with r5c4:
        swin   = st.number_input("Source Window (swin)", min_value=0.0, value=255.0, help="Source TCP window advertisement value")

    st.markdown("**Packet Size Means & Loss**")
    r6c1, r6c2, r6c3, r6c4 = st.columns(4)
    with r6c1:
        smean = st.number_input("Source Mean Pkt Size (smean)", min_value=0.0, value=43.0, help="Mean of flow packet size by source")
    with r6c2:
        dmean = st.number_input("Dest Mean Pkt Size (dmean)", min_value=0.0, value=43.0, help="Mean of flow packet size by destination")
    with r6c3:
        sloss = st.number_input("Source Loss (sloss)", min_value=0.0, value=0.0, help="Source packets retransmitted or dropped")
    with r6c4:
        dloss = st.number_input("Dest Loss (dloss)", min_value=0.0, value=0.0, help="Destination packets retransmitted or dropped")

    st.markdown("**Connection Tracking**")
    r7c1, r7c2, r7c3 = st.columns(3)
    with r7c1:
        ct_src_dport_ltm = st.number_input("Src-DstPort Connections (ct_src_dport_ltm)", min_value=0.0, value=1.0, help="Connections from same src IP and dst port in last 100 records")
    with r7c2:
        ct_dst_sport_ltm = st.number_input("Dst-SrcPort Connections (ct_dst_sport_ltm)", min_value=0.0, value=1.0, help="Connections to same dst IP and src port in last 100 records")
    with r7c3:
        is_sm_ips_ports  = st.number_input("Same Src/Dst IPs & Ports (is_sm_ips_ports)", min_value=0.0, max_value=1.0, value=0.0, help="1 if src/dst IPs and ports are equal, else 0")

    st.markdown("---")
    submitted = st.form_submit_button("🔍  ANALYZE TRAFFIC", use_container_width=True)

# ─────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────
if submitted:
    # Build raw input dict
    raw_input = {
        'proto':            proto_sel,
        'service':          service_sel,
        'state':            state_sel,
        'sbytes':           sbytes,
        'dbytes':           dbytes,
        'spkts':            spkts,
        'dpkts':            dpkts,
        'rate':             rate,
        'sload':            sload,
        'dload':            dload,
        'dur':              dur,
        'sinpkt':           sinpkt,
        'dinpkt':           dinpkt,
        'sjit':             sjit,
        'djit':             djit,
        'tcprtt':           tcprtt,
        'synack':           synack,
        'ackdat':           ackdat,
        'swin':             swin,
        'smean':            smean,
        'dmean':            dmean,
        'sloss':            sloss,
        'dloss':            dloss,
        'ct_src_dport_ltm': ct_src_dport_ltm,
        'ct_dst_sport_ltm': ct_dst_sport_ltm,
        'is_sm_ips_ports':  is_sm_ips_ports,
    }

    # Encode categoricals
    for col in ['proto', 'service', 'state']:
        le  = label_encoders[col]
        val = raw_input[col]
        if val in le.classes_:
            raw_input[col] = int(le.transform([val])[0])
        else:
            raw_input[col] = 0

    # Build DataFrame in exact feature order
    input_df = pd.DataFrame([raw_input], columns=SELECTED_FEATURES)

    # Scale
    input_scaled = scaler.transform(input_df)

    # Predict binary (attack or not)
    binary_pred   = xgb_binary.predict(input_scaled)[0]
    binary_proba  = xgb_binary.predict_proba(input_scaled)[0]
    confidence    = float(np.max(binary_proba)) * 100

    # Predict attack type
    multi_pred    = xgb_multiclass.predict(input_scaled)[0]
    attack_type   = target_encoder.inverse_transform([multi_pred])[0]

    # Risk score
    base_risk     = RISK_SCORES.get(attack_type, 50)
    risk_score    = int(base_risk * (confidence / 100))
    risk_label, risk_color = get_risk_level(risk_score)

    # ── RESULTS ──────────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-header">// Detection Results</p>', unsafe_allow_html=True)

    if binary_pred == 0:
        st.markdown(f"""
        <div class="result-normal">
            <div class="result-label" style="color:#3fb950;">✅ NORMAL TRAFFIC</div>
            <div style="color:#8b949e; margin-top:8px; font-size:0.95rem;">No attack detected in this network flow</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-attack">
            <div class="result-label" style="color:#ff7b72;">⚠️ ATTACK DETECTED</div>
            <div class="attack-type-badge">{attack_type.upper()}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Metrics row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f'<div class="metric-box"><span class="metric-value" style="color:#00e5ff;">{attack_type}</span><span class="metric-label">Attack Type</span></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-box"><span class="metric-value" style="color:#00e5ff;">{confidence:.1f}%</span><span class="metric-label">Confidence</span></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-box"><span class="metric-value" style="color:{risk_color};">{risk_score}/100</span><span class="metric-label">Risk Score</span></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="metric-box"><span class="metric-value" style="color:{risk_color};">{risk_label}</span><span class="metric-label">Risk Level</span></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Risk progress bar
    st.markdown("**Threat Level**")
    st.progress(risk_score / 100)

    # Recommendations
    st.markdown('<p class="section-header">// Recommended Actions</p>', unsafe_allow_html=True)
    if risk_score == 0:
        st.success("✅ System operating normally. No action required.")
    elif risk_score <= 30:
        st.info("🔵 Low threat detected. Monitor logs and review activity periodically.")
    elif risk_score <= 50:
        st.warning("🟡 Medium threat. Increase monitoring frequency and check for unauthorized access.")
    elif risk_score <= 75:
        st.warning("🟠 High threat! Investigate immediately. Consider isolating the affected system.")
    else:
        st.error("🔴 CRITICAL THREAT! Activate incident response. Isolate system and alert security team NOW.")

    # Attack info
    attack_info = {
        'Normal':         "Regular network traffic. No malicious activity.",
        'Reconnaissance': "Attacker is scanning/probing the network to gather information.",
        'Fuzzers':        "Random/unexpected data is being sent to crash or find vulnerabilities.",
        'Analysis':       "Deep packet inspection or port scanning activity detected.",
        'Generic':        "Generic attack not fitting other specific categories.",
        'DoS':            "Denial of Service attack — attempting to overwhelm the system.",
        'Exploits':       "Exploit attack targeting known software vulnerabilities.",
        'Backdoor':       "Backdoor access attempt — attacker trying to maintain hidden access.",
        'Shellcode':      "Shellcode injection detected — attempting to execute malicious code.",
        'Worms':          "Self-replicating malware spreading across the network.",
    }

    if attack_type in attack_info:
        st.markdown(f'<div class="tip-box">ℹ️ <b>{attack_type}:</b> {attack_info[attack_type]}</div>', unsafe_allow_html=True)

    # Debug expander
    with st.expander("🔍 View Input Summary"):
        st.dataframe(input_df.T.rename(columns={0: "Value"}), use_container_width=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<p style="text-align:center; color:#3d4450; font-size:0.8rem; font-family:\'Share Tech Mono\', monospace; letter-spacing:2px;">'
    'UNSW-NB15 DATASET · XGBOOST MODEL · CYBER ATTACK DETECTION SYSTEM'
    '</p>',
    unsafe_allow_html=True
)