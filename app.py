
# ═══════════════════════════════════════════════════════════════
#  Customer Churn Prediction Dashboard
#  E-commerce Customer Analytics
#  Run with: streamlit run app.py
# ═══════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import warnings
warnings.filterwarnings("ignore")

# ── Page Configuration ────────────────────────────────────────
st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS Styling ────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #2c3e50;
        text-align: center;
        padding: 1rem 0 0.5rem 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #7f8c8d;
        text-align: center;
        margin-bottom: 2rem;
    }
    .kpi-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: 800;
        margin: 0;
    }
    .kpi-label {
        font-size: 0.85rem;
        opacity: 0.9;
        margin: 0;
    }
    .risk-high   { color: #e74c3c; font-weight: bold; font-size: 1.3rem; }
    .risk-medium { color: #f39c12; font-weight: bold; font-size: 1.3rem; }
    .risk-low    { color: #2ecc71; font-weight: bold; font-size: 1.3rem; }
    .section-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #2c3e50;
        border-left: 4px solid #667eea;
        padding-left: 0.8rem;
        margin: 1.5rem 0 1rem 0;
    }
    div[data-testid="stMetricValue"] { font-size: 1.8rem; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# DATA & MODEL LOADING
# ════════════════════════════════════════════════════════════════

@st.cache_data   # Cache so data loads only once — not on every interaction
def load_data():
    df = pd.read_csv("predictions.csv")
    return df

@st.cache_resource   # Cache the model (heavier object)
def load_model():
    model     = joblib.load("churn_model.pkl")
    scaler    = joblib.load("scaler.pkl")
    threshold = joblib.load("best_threshold.pkl")
    feat_cols = joblib.load("feature_columns.pkl")
    return model, scaler, threshold, feat_cols

df                              = load_data()
model, scaler, threshold, cols  = load_model()

# ════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════

with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/shopping-cart.png", width=80)
    st.markdown("## 🛒 Churn Analytics")
    st.markdown("---")

    page = st.radio(
        "📌 Navigate to:",
        ["📊 Overview",
         "🔍 Segment Analysis",
         "👤 Customer Lookup",
         "💡 Retention Strategies"],
        index=0
    )

    st.markdown("---")
    st.markdown("### 🎯 Model Info")

    model_name = type(model).__name__
    st.info(f"""
    **Champion Model**  
    {model_name}  
    
    **Decision Threshold**  
    {threshold:.4f}
    """)

    st.markdown("---")
    st.markdown("### 📁 Dataset")
    st.success(f"""
    **Total Customers:** {len(df):,}  
    **Churned:** {df["Churn"].sum():,}  
    **Churn Rate:** {df["Churn"].mean()*100:.1f}%
    """)

    st.markdown("---")
    st.caption("Built with Streamlit + XGBoost + SHAP")


# ════════════════════════════════════════════════════════════════
# PAGE 1: OVERVIEW
# ════════════════════════════════════════════════════════════════

if page == "📊 Overview":

    st.markdown('<div class="main-header">🛒 Customer Churn Prediction Dashboard</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">E-commerce Customer Analytics — Real-time Churn Intelligence</div>',
                unsafe_allow_html=True)

    # ── KPI Cards ─────────────────────────────────────────────
    st.markdown('<div class="section-title">📌 Key Business Metrics</div>',
                unsafe_allow_html=True)

    total_customers  = len(df)
    total_churned    = int(df["churn_prediction"].sum())
    churn_rate       = df["churn_prediction"].mean() * 100
    avg_monthly      = df["MonthlyCharges"].mean()
    revenue_at_risk  = df[df["churn_prediction"]==1]["MonthlyCharges"].sum()
    high_risk_count  = (df["risk_segment"] == "🔴 High Risk").sum()

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        st.metric("👥 Total Customers",   f"{total_customers:,}")
    with col2:
        st.metric("🚨 Predicted Churners", f"{total_churned:,}",
                  delta=f"+{churn_rate:.1f}% rate", delta_color="inverse")
    with col3:
        st.metric("📉 Churn Rate",         f"{churn_rate:.1f}%")
    with col4:
        st.metric("💰 Avg Monthly Rev",    f"₹{avg_monthly:.0f}")
    with col5:
        st.metric("⚠️ Revenue at Risk",    f"₹{revenue_at_risk:,.0f}")
    with col6:
        st.metric("🔴 High Risk",          f"{high_risk_count:,}")

    st.markdown("---")

    # ── Charts Row 1 ──────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">Churn Distribution</div>',
                    unsafe_allow_html=True)
        churn_counts = df["Churn"].value_counts().reset_index()
        churn_counts.columns = ["Churn", "Count"]
        churn_counts["Label"] = churn_counts["Churn"].map({0: "Retained", 1: "Churned"})

        fig_pie = px.pie(
            churn_counts, values="Count", names="Label",
            color="Label",
            color_discrete_map={"Retained": "#2ecc71", "Churned": "#e74c3c"},
            hole=0.45
        )
        fig_pie.update_traces(textposition="outside", textinfo="percent+label",
                              textfont_size=13)
        fig_pie.update_layout(
            showlegend=True, height=350,
            annotations=[dict(text=f"{churn_rate:.0f}%<br>Churn",
                              x=0.5, y=0.5, font_size=16, showarrow=False)]
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        st.markdown('<div class="section-title">Risk Segment Breakdown</div>',
                    unsafe_allow_html=True)
        risk_counts = df["risk_segment"].value_counts().reset_index()
        risk_counts.columns = ["Segment", "Count"]

        fig_risk = px.bar(
            risk_counts, x="Segment", y="Count",
            color="Segment",
            color_discrete_map={
                "🟢 Low Risk"    : "#2ecc71",
                "🟡 Medium Risk" : "#f39c12",
                "🔴 High Risk"   : "#e74c3c"
            },
            text="Count"
        )
        fig_risk.update_traces(textposition="outside", textfont_size=12)
        fig_risk.update_layout(height=350, showlegend=False,
                               xaxis_title="Risk Segment",
                               yaxis_title="Number of Customers")
        st.plotly_chart(fig_risk, use_container_width=True)

    # ── Charts Row 2 ──────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">Churn Probability Distribution</div>',
                    unsafe_allow_html=True)
        fig_hist = px.histogram(
            df, x="churn_probability",
            color="risk_segment",
            color_discrete_map={
                "🟢 Low Risk"    : "#2ecc71",
                "🟡 Medium Risk" : "#f39c12",
                "🔴 High Risk"   : "#e74c3c"
            },
            nbins=40, opacity=0.8,
            labels={"churn_probability": "Predicted Churn Probability"}
        )
        fig_hist.add_vline(x=threshold, line_dash="dash",
                           line_color="black", line_width=2,
                           annotation_text=f"Threshold: {threshold:.2f}",
                           annotation_position="top right")
        fig_hist.update_layout(height=330, bargap=0.05)
        st.plotly_chart(fig_hist, use_container_width=True)

    with col2:
        st.markdown('<div class="section-title">Revenue at Risk by Segment</div>',
                    unsafe_allow_html=True)
        rev_by_segment = (df[df["churn_prediction"]==1]
                            .groupby("risk_segment")["MonthlyCharges"]
                            .sum().reset_index())
        rev_by_segment.columns = ["Segment", "Revenue at Risk"]

        fig_rev = px.bar(
            rev_by_segment, x="Segment", y="Revenue at Risk",
            color="Segment",
            color_discrete_map={
                "🟢 Low Risk"    : "#2ecc71",
                "🟡 Medium Risk" : "#f39c12",
                "🔴 High Risk"   : "#e74c3c"
            },
            text_auto=".2s"
        )
        fig_rev.update_layout(height=330, showlegend=False,
                              yaxis_title="Monthly Revenue at Risk (₹)")
        st.plotly_chart(fig_rev, use_container_width=True)


# ════════════════════════════════════════════════════════════════
# PAGE 2: SEGMENT ANALYSIS
# ════════════════════════════════════════════════════════════════

elif page == "🔍 Segment Analysis":

    st.markdown('<div class="main-header">🔍 Customer Segment Analysis</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Identify which customer groups are most at risk</div>',
                unsafe_allow_html=True)

    # ── Decode contract from one-hot back for display ──────────
    # Reconstruct readable labels from encoded columns
    def get_contract(row):
        if "Contract_One year" in row.index and row.get("Contract_One year", 0) == 1:
            return "One year"
        elif "Contract_Two year" in row.index and row.get("Contract_Two year", 0) == 1:
            return "Two year"
        else:
            return "Month-to-month"

    def get_internet(row):
        if "InternetService_Fiber optic" in row.index and row.get("InternetService_Fiber optic", 0) == 1:
            return "Fiber optic"
        elif "InternetService_No" in row.index and row.get("InternetService_No", 0) == 1:
            return "No Internet"
        else:
            return "DSL"

    df["Contract_Label"]  = df.apply(get_contract, axis=1)
    df["Internet_Label"]  = df.apply(get_internet, axis=1)
    df["Tenure_Bucket_Label"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 100],
        labels=["New (0-12m)", "Growing (13-24m)", "Loyal (25-48m)", "Champion (49m+)"]
    )

    # ── Row 1: Contract & Internet ─────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">Churn Rate by Contract Type</div>',
                    unsafe_allow_html=True)
        contract_churn = (df.groupby("Contract_Label")["Churn"]
                            .agg(["mean","count"]).reset_index())
        contract_churn.columns = ["Contract", "Churn Rate", "Count"]
        contract_churn["Churn Rate"] *= 100
        contract_churn = contract_churn.sort_values("Churn Rate", ascending=True)

        fig = px.bar(contract_churn, x="Churn Rate", y="Contract",
                     orientation="h", text="Churn Rate",
                     color="Churn Rate",
                     color_continuous_scale=["#2ecc71","#f39c12","#e74c3c"])
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(height=300, coloraxis_showscale=False,
                          xaxis_title="Churn Rate (%)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-title">Churn Rate by Internet Service</div>',
                    unsafe_allow_html=True)
        internet_churn = (df.groupby("Internet_Label")["Churn"]
                            .agg(["mean","count"]).reset_index())
        internet_churn.columns = ["Internet", "Churn Rate", "Count"]
        internet_churn["Churn Rate"] *= 100
        internet_churn = internet_churn.sort_values("Churn Rate", ascending=True)

        fig2 = px.bar(internet_churn, x="Churn Rate", y="Internet",
                      orientation="h", text="Churn Rate",
                      color="Churn Rate",
                      color_continuous_scale=["#2ecc71","#f39c12","#e74c3c"])
        fig2.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig2.update_layout(height=300, coloraxis_showscale=False,
                           xaxis_title="Churn Rate (%)")
        st.plotly_chart(fig2, use_container_width=True)

    # ── Row 2: Tenure Bucket & Charges ────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">Churn Rate by Tenure Stage</div>',
                    unsafe_allow_html=True)
        tenure_churn = (df.groupby("Tenure_Bucket_Label", observed=True)["Churn"]
                          .mean().reset_index())
        tenure_churn.columns = ["Tenure Stage", "Churn Rate"]
        tenure_churn["Churn Rate"] *= 100

        fig3 = px.bar(tenure_churn, x="Tenure Stage", y="Churn Rate",
                      text="Churn Rate", color="Churn Rate",
                      color_continuous_scale=["#2ecc71","#f39c12","#e74c3c"])
        fig3.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig3.update_layout(height=320, coloraxis_showscale=False,
                           yaxis_title="Churn Rate (%)")
        st.plotly_chart(fig3, use_container_width=True)

    with col2:
        st.markdown('<div class="section-title">Monthly Charges vs Churn Probability</div>',
                    unsafe_allow_html=True)
        sample_df = df.sample(min(500, len(df)), random_state=42)

        fig4 = px.scatter(
            sample_df, x="MonthlyCharges", y="churn_probability",
            color="risk_segment",
            color_discrete_map={
                "🟢 Low Risk"    : "#2ecc71",
                "🟡 Medium Risk" : "#f39c12",
                "🔴 High Risk"   : "#e74c3c"
            },
            opacity=0.6, size_max=6,
            labels={"churn_probability": "Churn Probability",
                    "MonthlyCharges": "Monthly Charges (₹)"}
        )
        fig4.add_hline(y=threshold, line_dash="dash",
                       line_color="black",
                       annotation_text=f"Decision Threshold ({threshold:.2f})")
        fig4.update_layout(height=320)
        st.plotly_chart(fig4, use_container_width=True)

    # ── Heatmap: Contract x Tenure ─────────────────────────────
    st.markdown('<div class="section-title">Churn Rate Heatmap — Contract × Tenure Stage</div>',
                unsafe_allow_html=True)

    heatmap_data = (df.groupby(["Contract_Label","Tenure_Bucket_Label"], observed=True)["Churn"]
                      .mean().reset_index())
    heatmap_data.columns = ["Contract", "Tenure Stage", "Churn Rate"]
    heatmap_pivot = heatmap_data.pivot(index="Contract",
                                        columns="Tenure Stage",
                                        values="Churn Rate") * 100

    fig5 = px.imshow(
        heatmap_pivot, text_auto=".1f",
        color_continuous_scale=["#2ecc71","#f1c40f","#e74c3c"],
        labels=dict(color="Churn Rate (%)"),
        aspect="auto"
    )
    fig5.update_layout(height=300,
                       title="Darker red = highest churn risk segment")
    st.plotly_chart(fig5, use_container_width=True)


# ════════════════════════════════════════════════════════════════
# PAGE 3: CUSTOMER LOOKUP
# ════════════════════════════════════════════════════════════════

elif page == "👤 Customer Lookup":

    st.markdown('<div class="main-header">👤 Individual Customer Risk Lookup</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Enter a Customer ID to see their churn risk profile</div>',
                unsafe_allow_html=True)

    # ── Search Bar ────────────────────────────────────────────
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        customer_id = st.selectbox(
            "🔍 Select or type a Customer ID:",
            options=df["CustomerID"].tolist(),
            index=0
        )

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        search_btn = st.button("🔎 Look Up Customer", type="primary")

    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        random_btn = st.button("🎲 Random High-Risk Customer")

    if random_btn:
        high_risk_df = df[df["risk_segment"] == "🔴 High Risk"]
        if len(high_risk_df) > 0:
            customer_id = high_risk_df.sample(1)["CustomerID"].values[0]
            st.success(f"🎲 Randomly selected high-risk customer: **{customer_id}**")

    # ── Customer Profile ──────────────────────────────────────
    customer = df[df["CustomerID"] == customer_id].iloc[0]
    churn_prob = customer["churn_probability"]
    risk_seg   = customer["risk_segment"]

    st.markdown("---")

    # Risk indicator
    if risk_seg == "🔴 High Risk":
        risk_color = "#e74c3c"
        risk_msg   = "⚠️ IMMEDIATE ATTENTION REQUIRED"
    elif risk_seg == "🟡 Medium Risk":
        risk_color = "#f39c12"
        risk_msg   = "👀 MONITOR CLOSELY"
    else:
        risk_color = "#2ecc71"
        risk_msg   = "✅ HEALTHY CUSTOMER"

    st.markdown(f"""
    <div style="background:{risk_color}20; border-left:5px solid {risk_color};
                padding:1rem; border-radius:8px; margin-bottom:1rem;">
        <h3 style="color:{risk_color}; margin:0;">{risk_seg} — {risk_msg}</h3>
        <p style="margin:0.3rem 0 0 0; font-size:1.1rem;">
            Customer <b>{customer_id}</b> has a
            <b style="color:{risk_color};">{churn_prob*100:.1f}%</b> probability of churning
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI row ────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🎯 Churn Probability", f"{churn_prob*100:.1f}%")
    col2.metric("📅 Tenure",            f"{int(customer['tenure'])} months")
    col3.metric("💳 Monthly Charges",   f"₹{customer['MonthlyCharges']:.2f}")
    col4.metric("💰 Total Charges",     f"₹{customer['TotalCharges']:.2f}")

    st.markdown("---")

    # ── Gauge Chart ───────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">Churn Risk Gauge</div>',
                    unsafe_allow_html=True)
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=churn_prob * 100,
            number={"suffix": "%", "font": {"size": 36}},
            delta={"reference": threshold * 100,
                   "decreasing": {"color": "#2ecc71"},
                   "increasing": {"color": "#e74c3c"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar":  {"color": risk_color, "thickness": 0.3},
                "steps": [
                    {"range": [0,  30],  "color": "#d5f5e3"},
                    {"range": [30, 60],  "color": "#fef9e7"},
                    {"range": [60, 100], "color": "#fadbd8"},
                ],
                "threshold": {
                    "line":  {"color": "black", "width": 4},
                    "thickness": 0.8,
                    "value": threshold * 100
                }
            },
            title={"text": "Churn Probability", "font": {"size": 18}}
        ))
        fig_gauge.update_layout(height=300, margin=dict(t=60, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col2:
        st.markdown('<div class="section-title">Top Churn Risk Factors</div>',
                    unsafe_allow_html=True)

        # Show top contributing features based on feature values
        # (Simplified SHAP proxy — actual SHAP needs the explainer here)
        feature_display = {
            "Contract Type"       : "Month-to-month" if customer.get("Contract_One year",0)==0
                                    and customer.get("Contract_Two year",0)==0 else "Annual+",
            "Internet Service"    : "Fiber optic" if customer.get("InternetService_Fiber optic",0)==1
                                    else "DSL/None",
            "Online Security"     : "No" if customer.get("OnlineSecurity",0)==0 else "Yes",
            "Tech Support"        : "No" if customer.get("TechSupport",0)==0 else "Yes",
            "Tenure (months)"     : int(customer["tenure"]),
            "Monthly Charges"     : f"₹{customer['MonthlyCharges']:.2f}",
            "Number of Services"  : int(customer.get("num_services", 0)),
            "Paperless Billing"   : "Yes" if customer.get("PaperlessBilling",0)==1 else "No"
        }

        for feat, val in feature_display.items():
            col_a, col_b = st.columns([2, 1])
            col_a.write(f"**{feat}**")
            col_b.write(f"`{val}`")

    st.markdown("---")

    # ── Comparison vs Average ──────────────────────────────────
    st.markdown('<div class="section-title">📊 This Customer vs. Population Average</div>',
                unsafe_allow_html=True)

    compare_cols = ["tenure", "MonthlyCharges", "TotalCharges",
                    "num_services", "churn_probability"]
    compare_labels = ["Tenure (months)", "Monthly Charges (₹)",
                      "Total Charges (₹)", "# Services", "Churn Probability"]

    cust_vals = [customer[c] for c in compare_cols]
    avg_vals  = [df[c].mean() for c in compare_cols]

    fig_compare = go.Figure()
    fig_compare.add_trace(go.Bar(
        name="This Customer", x=compare_labels, y=cust_vals,
        marker_color=risk_color, opacity=0.85
    ))
    fig_compare.add_trace(go.Bar(
        name="Population Average", x=compare_labels, y=avg_vals,
        marker_color="#95a5a6", opacity=0.85
    ))
    fig_compare.update_layout(
        barmode="group", height=350,
        yaxis_title="Value",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_compare, use_container_width=True)


# ════════════════════════════════════════════════════════════════
# PAGE 4: RETENTION STRATEGIES
# ════════════════════════════════════════════════════════════════

elif page == "💡 Retention Strategies":

    st.markdown('<div class="main-header">💡 Retention Strategy Recommender</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-powered retention actions based on churn drivers</div>',
                unsafe_allow_html=True)

    # ── Priority Queue: Top At-Risk Customers ─────────────────
    st.markdown('<div class="section-title">🚨 Top 10 Highest-Risk Customers — Act Now</div>',
                unsafe_allow_html=True)

    top_risk = (df[df["churn_prediction"]==1]
                  .sort_values("churn_probability", ascending=False)
                  .head(10)[[
                      "CustomerID","churn_probability","risk_segment",
                      "MonthlyCharges","tenure","TotalCharges"
                  ]]
                  .copy())

    top_risk["churn_probability"] = (top_risk["churn_probability"]*100).round(1).astype(str) + "%"
    top_risk["MonthlyCharges"]    = "₹" + top_risk["MonthlyCharges"].round(2).astype(str)
    top_risk["TotalCharges"]      = "₹" + top_risk["TotalCharges"].round(2).astype(str)
    top_risk["tenure"]            = top_risk["tenure"].astype(int).astype(str) + " months"
    top_risk.columns              = ["Customer ID","Churn Probability","Risk Segment",
                                     "Monthly Rev","Tenure","Total Rev"]

    st.dataframe(top_risk, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Strategy Cards ────────────────────────────────────────
    st.markdown('<div class="section-title">📋 Retention Playbook by Risk Segment</div>',
                unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.error("🔴 HIGH RISK STRATEGY")
        st.markdown("""
        **Target:** Month-to-month + High Charges + Low Tenure

        **Actions:**
        - 📞 Personal call from retention team within 24 hours
        - 💳 Offer 3-month discount (20-30% off)
        - 🔒 Propose annual contract with free perks
        - 🎁 Complimentary add-on: Security + TechSupport bundle
        - 📊 Share personalized usage insights report

        **Goal:** Convert to annual contract
        **Budget:** Up to ₹500 per customer
        """)

    with col2:
        st.warning("🟡 MEDIUM RISK STRATEGY")
        st.markdown("""
        **Target:** Growing tenure + Moderate charges

        **Actions:**
        - 📧 Personalized email with loyalty offer
        - 🌟 Introduce loyalty points / rewards program
        - 📦 Bundle upgrade at current price
        - 🔧 Proactive tech support check-in
        - 📱 Push notification: feature they haven't used

        **Goal:** Increase engagement & service usage
        **Budget:** Up to ₹150 per customer
        """)

    with col3:
        st.success("🟢 LOW RISK STRATEGY")
        st.markdown("""
        **Target:** Long tenure + Annual contract

        **Actions:**
        - 🏆 VIP / Champion loyalty recognition
        - 📣 Referral program invitation
        - 🆕 Early access to new features
        - 💌 Quarterly satisfaction survey
        - 🎂 Anniversary rewards on tenure milestones

        **Goal:** Maintain loyalty & generate referrals
        **Budget:** Up to ₹50 per customer
        """)

    st.markdown("---")

    # ── Expected ROI Calculator ────────────────────────────────
    st.markdown('<div class="section-title">💰 Retention Campaign ROI Estimator</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        intervention_rate = st.slider(
            "📊 % of predicted churners we intervene with:",
            min_value=10, max_value=100, value=50, step=10
        )
        success_rate = st.slider(
            "🎯 Estimated retention success rate:",
            min_value=5, max_value=50, value=20, step=5
        )
        avg_campaign_cost = st.number_input(
            "💸 Average campaign cost per customer (₹):",
            min_value=50, max_value=1000, value=200, step=50
        )

    with col2:
        total_predicted_churners = int(df["churn_prediction"].sum())
        customers_intervened     = int(total_predicted_churners * intervention_rate / 100)
        customers_saved          = int(customers_intervened * success_rate / 100)
        avg_monthly_rev          = df["MonthlyCharges"].mean()
        revenue_recovered        = customers_saved * avg_monthly_rev * 12  # annual
        total_campaign_cost      = customers_intervened * avg_campaign_cost
        net_roi                  = revenue_recovered - total_campaign_cost
        roi_pct                  = (net_roi / total_campaign_cost * 100) if total_campaign_cost > 0 else 0

        st.markdown(f"""
        ### 📊 Estimated Campaign Outcomes:

        | Metric | Value |
        |--------|-------|
        | Predicted Churners | {total_predicted_churners:,} |
        | Customers Intervened | {customers_intervened:,} |
        | Customers Saved | {customers_saved:,} |
        | Annual Revenue Recovered | ₹{revenue_recovered:,.0f} |
        | Total Campaign Cost | ₹{total_campaign_cost:,.0f} |
        | **Net ROI** | **₹{net_roi:,.0f}** |
        | **ROI %** | **{roi_pct:.0f}%** |
        """)

        if net_roi > 0:
            st.success(f"✅ Positive ROI! Campaign recovers ₹{net_roi:,.0f} net annually.")
        else:
            st.error("❌ Negative ROI — adjust campaign cost or target a smaller group.")

    st.markdown("---")
    st.markdown('<div class="section-title">📥 Download Churn Report</div>',
                unsafe_allow_html=True)

    csv_export = df[[
        "CustomerID","churn_probability","churn_prediction",
        "risk_segment","MonthlyCharges","tenure","TotalCharges"
    ]].copy()
    csv_export["churn_probability"] = (csv_export["churn_probability"]*100).round(2)
    csv_export.columns = [
        "Customer ID","Churn Probability (%)","Churn Predicted",
        "Risk Segment","Monthly Charges","Tenure (months)","Total Charges"
    ]

    st.download_button(
        label="📥 Download Full Churn Report (CSV)",
        data=csv_export.to_csv(index=False).encode("utf-8"),
        file_name="churn_report.csv",
        mime="text/csv",
        type="primary"
    )
