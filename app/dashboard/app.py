"""Interactive Streamlit AML Compliance Officer & Transaction Monitoring Dashboard.

Connects seamlessly to the FastAPI backend (http://localhost:8000) for real-time
inference and dual-granularity Explainable AI (SHAP & LIME).
"""

import json
import time
from typing import Any, Dict, List, Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# -----------------------------------------------------------------------------
# Configuration & Theme Settings
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AML Monitoring & XAI Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished financial compliance styling
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        border-left: 5px solid #1f77b4;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 12px;
    }
    .badge-suspicious {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1rem;
        display: inline-block;
        border: 1px solid #f87171;
    }
    .badge-legitimate {
        background-color: #dcfce7;
        color: #166534;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1rem;
        display: inline-block;
        border: 1px solid #4ade80;
    }
    .info-box {
        background-color: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 8px;
        padding: 14px;
        font-size: 0.95rem;
        color: #1e3a8a;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Sidebar: Backend Connection & Server Status
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image(
        "https://img.shields.io/badge/AML-Transaction%20Monitoring-blue?style=for-the-badge&logo=shield",
        use_container_width=True,
    )
    st.title("Server Connection")
    api_url = st.text_input("FastAPI Endpoint URL", value="http://localhost:8000")

    # Health check probe
    backend_online = False
    health_info: Dict[str, Any] = {}
    try:
        resp = requests.get(f"{api_url}/health", timeout=3.0)
        if resp.status_code == 200:
            backend_online = True
            health_info = resp.json()
    except Exception:
        backend_online = False

    if backend_online:
        st.success("● FastAPI Connected", icon="🟢")
        st.markdown(f"**Model:** `{health_info.get('model_name', 'Active')}`")
        st.markdown(f"**Status:** `{health_info.get('status', 'healthy')}`")
        if health_info.get("threshold") is not None:
            st.markdown(f"**Threshold:** `{health_info['threshold']:.4f}`")
        if health_info.get("feature_count") is not None:
            st.markdown(f"**Features:** `{health_info['feature_count']:,}`")
    else:
        st.error("● FastAPI Offline", icon="🔴")
        st.caption(
            "Start the API server using:\n```bash\nuvicorn src.api:app --reload\n```"
        )

    st.divider()
    st.markdown("### Quick Navigation")
    st.caption("Anti-Money Laundering transaction risk scoring and regulatory transparency audit.")

    if st.button("🔄 Hot Reload Model Artifacts", use_container_width=True):
        try:
            r = requests.post(f"{api_url}/model/reload", timeout=5.0)
            if r.status_code == 200:
                st.toast("Model artifacts reloaded successfully!", icon="✅")
                st.rerun()
            else:
                st.error(f"Reload failed: {r.text}")
        except Exception as e:
            st.error(f"Cannot connect to API: {e}")


# -----------------------------------------------------------------------------
# Main Application Header
# -----------------------------------------------------------------------------
st.title("🛡️ AML Transaction Monitoring & Explainable AI")
st.markdown(
    "Real-time financial crime detection, dynamic risk scoring, and audit-ready "
    "**SHAP / LIME** attributions for regulatory transparency (FATF, FinCEN & EU AI Act)."
)

tab_single, tab_batch, tab_governance = st.tabs(
    ["🔍 Single Transaction Audit", "📁 Batch Pipeline Audit", "📊 Model Governance & Diagnostic"]
)

# =============================================================================
# TAB 1: SINGLE TRANSACTION AUDIT
# =============================================================================
with tab_single:
    st.subheader("Evaluate Incoming Transaction")

    # Sample presets
    col_preset1, col_preset2, col_preset3 = st.columns([1, 1, 2])
    with col_preset1:
        load_suspicious = st.button("🚨 Load Suspicious Cross-Border Sample")
    with col_preset2:
        load_normal = st.button("✅ Load Everyday Retail Sample")

    # Default preset values
    default_sender = "ACC-891024"
    default_receiver = "ACC-381940"
    default_amount = 45000.0
    default_pay_curr = "US Dollar"
    default_rec_curr = "Euro"
    default_send_loc = "USA"
    default_rec_loc = "Switzerland"
    default_payment_type = "Cross-border"
    default_date = "2023-10-15"
    default_time = "23:45:10"

    if load_suspicious:
        default_sender = "ACC-SUSP-7721"
        default_receiver = "ACC-SHELL-9901"
        default_amount = 185000.0
        default_pay_curr = "US Dollar"
        default_rec_curr = "Dirham"
        default_send_loc = "USA"
        default_rec_loc = "UAE"
        default_payment_type = "Cross-border"
        default_date = "2023-11-01"
        default_time = "02:15:30"
    elif load_normal:
        default_sender = "ACC-RETAIL-1002"
        default_receiver = "ACC-GROCERY-4011"
        default_amount = 85.50
        default_pay_curr = "UK pounds"
        default_rec_curr = "UK pounds"
        default_send_loc = "UK"
        default_rec_loc = "UK"
        default_payment_type = "Debit card"
        default_date = "2023-11-01"
        default_time = "14:30:00"

    # Input Form
    with st.form("tx_input_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("##### Account Identifiers")
            sender_account = st.text_input("Sender Account", value=default_sender)
            receiver_account = st.text_input("Receiver Account", value=default_receiver)
            amount = st.number_input(
                "Amount", min_value=0.01, max_value=1e8, value=float(default_amount), step=100.0
            )

        with col2:
            st.markdown("##### Currencies & Locations")
            currencies = [
                "US Dollar", "Euro", "UK pounds", "Dirham", "Indian rupee",
                "Swiss franc", "Mexican Peso", "Moroccan dirham", "Naira",
                "Pakistani rupee", "Turkish lira", "Yen", "Albanian lek"
            ]
            payment_currency = st.selectbox(
                "Payment Currency", currencies,
                index=currencies.index(default_pay_curr) if default_pay_curr in currencies else 0
            )
            received_currency = st.selectbox(
                "Received Currency", currencies,
                index=currencies.index(default_rec_curr) if default_rec_curr in currencies else 1
            )
            locations = [
                "USA", "Switzerland", "UK", "UAE", "Germany", "France", "Spain",
                "Italy", "India", "Pakistan", "Mexico", "Morocco", "Nigeria",
                "Turkey", "Japan", "Netherlands", "Austria", "Albania"
            ]
            sender_location = st.selectbox(
                "Sender Bank Location", locations,
                index=locations.index(default_send_loc) if default_send_loc in locations else 0
            )
            receiver_location = st.selectbox(
                "Receiver Bank Location", locations,
                index=locations.index(default_rec_loc) if default_rec_loc in locations else 1
            )

        with col3:
            st.markdown("##### Transfer Method & Temporal")
            payment_types = [
                "Cross-border", "Cash Deposit", "Cash Withdrawal", "Cheque",
                "Credit card", "Debit card", "ACH"
            ]
            payment_type = st.selectbox(
                "Payment Type", payment_types,
                index=payment_types.index(default_payment_type) if default_payment_type in payment_types else 0
            )
            tx_date = st.text_input("Transaction Date (YYYY-MM-DD)", value=default_date)
            tx_time = st.text_input("Transaction Time (HH:MM:SS)", value=default_time)

            st.markdown("##### Explainability Settings")
            xai_method = st.radio("XAI Method", ["shap", "lime"], index=0, horizontal=True)
            top_k = st.slider("Top Attributions", min_value=3, max_value=12, value=5)

        submitted = st.form_submit_button("⚡ Evaluate Transaction & Explain", use_container_width=True)

    if submitted:
        if not backend_online:
            st.error("Cannot connect to FastAPI backend at " + api_url + ". Please start the server.")
        else:
            payload = {
                "Sender_account": sender_account,
                "Receiver_account": receiver_account,
                "Amount": amount,
                "Payment_currency": payment_currency,
                "Received_currency": received_currency,
                "Sender_bank_location": sender_location,
                "Receiver_bank_location": receiver_location,
                "Payment_type": payment_type,
                "Date": tx_date,
                "Time": tx_time,
            }

            with st.spinner("Executing real-time inference & feature attribution..."):
                t0 = time.time()
                try:
                    res_pred = requests.post(f"{api_url}/predict", json=payload, timeout=10.0)
                    pred_data = res_pred.json()
                except Exception as e:
                    pred_data = {"error": str(e)}

                try:
                    res_xai = requests.post(
                        f"{api_url}/explain?method={xai_method}&top_k={top_k}",
                        json=payload,
                        timeout=35.0,
                    )
                    xai_data = res_xai.json() if res_xai.status_code == 200 else None
                except Exception:
                    xai_data = None

                elapsed_ms = round((time.time() - t0) * 1000, 1)

            # Display Results
            if "error" in pred_data:
                st.error(f"Prediction failed: {pred_data['error']}")
            else:
                st.markdown("---")
                st.subheader("Inference Result")

                is_susp = pred_data.get("is_suspicious", False)
                risk_score = pred_data.get("risk_score", 0.0)
                risk_level = pred_data.get("risk_level", "Low")
                pred_label = pred_data.get("prediction_label", "Legitimate")

                res_col1, res_col2, res_col3, res_col4 = st.columns(4)

                with res_col1:
                    st.markdown("**Compliance Decision**")
                    if is_susp:
                        st.markdown("<span class='badge-suspicious'>🚨 SUSPICIOUS (ALERT)</span>", unsafe_allow_html=True)
                    else:
                        st.markdown("<span class='badge-legitimate'>✅ LEGITIMATE (PASS)</span>", unsafe_allow_html=True)

                with res_col2:
                    st.metric(
                        "Risk Score",
                        f"{risk_score:.4f}",
                        delta=f"Level: {risk_level}",
                        delta_color="inverse" if is_susp else "normal",
                    )

                with res_col3:
                    st.metric("Model Classification", pred_label)

                with res_col4:
                    st.metric("Total Latency", f"{elapsed_ms} ms", f"API: {pred_data.get('processing_time_ms', 0)} ms")

                # Progress gauge
                st.progress(min(max(float(risk_score), 0.0), 1.0))

                # Explainability Section
                st.markdown("### 🔍 Explainable AI (XAI) Attribution Breakdown")
                if xai_data:
                    domain_contribs = xai_data.get("domain_contributions", [])
                    raw_contribs = xai_data.get("raw_contributions", [])

                    col_chart, col_table = st.columns([3, 2])

                    with col_chart:
                        st.markdown("#### High-Level Domain Impact")
                        if domain_contribs:
                            df_domain = pd.DataFrame(domain_contribs)
                            df_domain["Net Impact"] = df_domain["net_attribution"].astype(float)
                            df_domain["Feature"] = df_domain["domain_feature"]

                            fig = px.bar(
                                df_domain,
                                x="Net Impact",
                                y="Feature",
                                orientation="h",
                                color="direction",
                                color_discrete_map={
                                    "suspicious_pusher": "#ef4444",
                                    "legitimacy_anchor": "#22c55e",
                                    "neutral": "#94a3b8",
                                },
                                title=f"{xai_method.upper()} Domain Feature Contributions",
                            )
                            fig.update_layout(yaxis=dict(autorange="reversed"), height=350)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("No domain aggregations returned.")

                    with col_table:
                        st.markdown("#### Top Model Attributions")
                        if raw_contribs:
                            df_raw = pd.DataFrame(raw_contribs)[
                                ["feature_name", "attribution", "direction"]
                            ]
                            st.dataframe(
                                df_raw.style.map(
                                    lambda val: "color: red; font-weight: bold"
                                    if val == "suspicious_pusher"
                                    else ("color: green; font-weight: bold" if val == "legitimacy_anchor" else ""),
                                    subset=["direction"],
                                ),
                                height=350,
                                use_container_width=True,
                            )
                        else:
                            st.info("No raw attributions returned.")

                    if "disclaimer" in xai_data:
                        st.caption(f"ℹ️ {xai_data['disclaimer']}")
                else:
                    st.warning("Explainability output was not returned or timed out.")


# =============================================================================
# TAB 2: BATCH PIPELINE AUDIT
# =============================================================================
with tab_batch:
    st.subheader("Batch Transaction Risk Scoring")
    st.markdown("Audit multi-record transaction feeds simultaneously via `/predict/batch`.")

    sample_batch = [
        {"Sender_account": "ACC-1001", "Receiver_account": "ACC-2001", "Amount": 1200.0, "Payment_type": "ACH", "Payment_currency": "US Dollar", "Received_currency": "US Dollar", "Sender_bank_location": "USA", "Receiver_bank_location": "USA"},
        {"Sender_account": "ACC-1002", "Receiver_account": "ACC-2002", "Amount": 85000.0, "Payment_type": "Cross-border", "Payment_currency": "US Dollar", "Received_currency": "Euro", "Sender_bank_location": "USA", "Receiver_bank_location": "Switzerland"},
        {"Sender_account": "ACC-1003", "Receiver_account": "ACC-2003", "Amount": 4900.0, "Payment_type": "Cash Deposit", "Payment_currency": "UK pounds", "Received_currency": "UK pounds", "Sender_bank_location": "UK", "Receiver_bank_location": "UK"},
        {"Sender_account": "ACC-1004", "Receiver_account": "ACC-2004", "Amount": 195000.0, "Payment_type": "Cross-border", "Payment_currency": "US Dollar", "Received_currency": "Dirham", "Sender_bank_location": "USA", "Receiver_bank_location": "UAE"},
        {"Sender_account": "ACC-1005", "Receiver_account": "ACC-2005", "Amount": 45.0, "Payment_type": "Debit card", "Payment_currency": "Euro", "Received_currency": "Euro", "Sender_bank_location": "Germany", "Receiver_bank_location": "Germany"},
    ]

    batch_input_str = st.text_area(
        "Enter JSON array of transactions",
        value=json.dumps(sample_batch, indent=2),
        height=220,
    )

    if st.button("🚀 Process Batch Audit", use_container_width=True):
        if not backend_online:
            st.error("FastAPI backend is offline.")
        else:
            try:
                tx_list = json.loads(batch_input_str)
                with st.spinner("Scoring batch transactions..."):
                    resp_b = requests.post(f"{api_url}/predict/batch", json={"transactions": tx_list}, timeout=15.0)
                
                if resp_b.status_code == 200:
                    b_data = resp_b.json()
                    st.success(f"Processed {b_data['total_processed']} transactions in {b_data['processing_time_ms']} ms")

                    # Summary Metrics
                    b_col1, b_col2, b_col3 = st.columns(3)
                    susp_cnt = b_data["suspicious_count"]
                    tot_cnt = b_data["total_processed"]
                    rate = (susp_cnt / max(tot_cnt, 1)) * 100.0

                    b_col1.metric("Total Processed", tot_cnt)
                    b_col2.metric("Suspicious Alerts", susp_cnt)
                    b_col3.metric("Suspicion Rate", f"{rate:.1f}%")

                    # Results Table
                    records = []
                    for r in b_data["results"]:
                        records.append({
                            "Account ID": r["transaction_id"],
                            "Suspicious": "🚨 YES" if r["is_suspicious"] else "✅ No",
                            "Risk Score": r["risk_score"],
                            "Risk Level": r["risk_level"],
                            "Classification": r["prediction_label"],
                            "Latency (ms)": r["processing_time_ms"],
                        })
                    df_res = pd.DataFrame(records)
                    st.dataframe(df_res, use_container_width=True)
                else:
                    st.error(f"Batch request failed: {resp_b.text}")
            except Exception as ex:
                st.error(f"Failed parsing or sending batch: {ex}")


# =============================================================================
# TAB 3: MODEL GOVERNANCE & DIAGNOSTIC
# =============================================================================
with tab_governance:
    st.subheader("Model Diagnostic & Governance Report")

    st.markdown(
        """
        <div class="info-box">
        <b>Audit Note on Model Training & Typology Features:</b><br>
        The active model (<code>XGBoost_No_SMOTE</code>) was trained on the SAML-D dataset.
        During training, the <code>Laundering_type</code> field (which labels specific crime typologies such as Structuring,
        Smurfing, or Normal cash movements) was included in the one-hot encoding feature matrix.
        In live production, incoming transactions do not have known typologies. The system safely defaults these to
        <code>Unknown</code> and operates against an empirically calibrated alert threshold.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Active Configuration")
    if backend_online and health_info:
        st.json(health_info)
    else:
        st.info("Backend is currently offline.")
