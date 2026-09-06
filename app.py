import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

# Plotly is preferred for charts, but the app must still work if it
# isn't installed — fall back to Altair (bundled with Streamlit).
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False

import altair as alt

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Financial Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# COLOR THEME
# ============================================================

COLOR_PRIMARY = "#2563EB"
COLOR_SECONDARY = "#7C3AED"
COLOR_SUCCESS = "#16A34A"
COLOR_DANGER = "#DC2626"
COLOR_WARNING = "#F59E0B"
COLOR_BG_DARK = "#0F172A"
COLOR_CARD_BG = "#1E293B"
COLOR_TEXT_LIGHT = "#F8FAFC"
COLOR_TEXT_MUTED = "#94A3B8"

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    f"""
    <style>
        .stApp {{
            background-color: {COLOR_BG_DARK};
            color: {COLOR_TEXT_LIGHT};
        }}

        section[data-testid="stSidebar"] {{
            background-color: {COLOR_CARD_BG};
            border-right: 1px solid rgba(148, 163, 184, 0.15);
        }}

        h1, h2, h3, h4, h5, h6 {{
            color: {COLOR_TEXT_LIGHT} !important;
        }}

        p, span, label, div {{
            color: {COLOR_TEXT_LIGHT};
        }}

        .app-caption {{
            color: {COLOR_TEXT_MUTED} !important;
        }}

        /* --- Generic card --- */
        .kpi-card {{
            background-color: {COLOR_CARD_BG};
            border-radius: 14px;
            padding: 22px 20px;
            border: 1px solid rgba(148, 163, 184, 0.12);
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);
            height: 100%;
        }}

        .kpi-icon {{
            font-size: 26px;
            margin-bottom: 6px;
        }}

        .kpi-label {{
            font-size: 13px;
            font-weight: 600;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            color: {COLOR_TEXT_MUTED};
            margin-bottom: 6px;
        }}

        .kpi-value {{
            font-size: 30px;
            font-weight: 800;
            line-height: 1.1;
            margin-bottom: 4px;
        }}

        .kpi-sub {{
            font-size: 13px;
            color: {COLOR_TEXT_MUTED};
        }}

        /* --- Section header --- */
        .section-header {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin: 8px 0 2px 0;
        }}

        .section-header h3 {{
            margin: 0;
        }}

        .section-sub {{
            color: {COLOR_TEXT_MUTED};
            font-size: 14px;
            margin-bottom: 14px;
        }}

        /* --- Info card (used for preprocessing / model info) --- */
        .info-card {{
            background-color: {COLOR_CARD_BG};
            border-radius: 14px;
            padding: 18px 20px;
            border-left: 4px solid {COLOR_PRIMARY};
            border-top: 1px solid rgba(148, 163, 184, 0.1);
            border-right: 1px solid rgba(148, 163, 184, 0.1);
            border-bottom: 1px solid rgba(148, 163, 184, 0.1);
            margin-bottom: 12px;
            height: 100%;
        }}

        .info-card h4 {{
            margin: 0 0 8px 0;
            font-size: 15px;
        }}

        .info-card ul {{
            margin: 0;
            padding-left: 18px;
            color: {COLOR_TEXT_MUTED};
            font-size: 14px;
        }}

        /* --- Risk result cards --- */
        .risk-card {{
            border-radius: 16px;
            padding: 28px;
            text-align: center;
            margin: 10px 0 18px 0;
        }}

        .risk-card-fraud {{
            background: linear-gradient(135deg, rgba(220,38,38,0.18), rgba(220,38,38,0.05));
            border: 1px solid rgba(220,38,38,0.5);
        }}

        .risk-card-legit {{
            background: linear-gradient(135deg, rgba(22,163,74,0.18), rgba(22,163,74,0.05));
            border: 1px solid rgba(22,163,74,0.5);
        }}

        .risk-card-title {{
            font-size: 22px;
            font-weight: 800;
            margin-bottom: 6px;
        }}

        .risk-card-prob {{
            font-size: 44px;
            font-weight: 900;
            margin: 6px 0;
        }}

        .risk-card-note {{
            color: {COLOR_TEXT_MUTED};
            font-size: 14px;
        }}

        /* --- Badges --- */
        .badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 0.03em;
        }}

        .badge-low {{ background: rgba(22,163,74,0.18); color: {COLOR_SUCCESS}; }}
        .badge-moderate {{ background: rgba(245,158,11,0.18); color: {COLOR_WARNING}; }}
        .badge-high {{ background: rgba(220,38,38,0.15); color: #F97316; }}
        .badge-veryhigh {{ background: rgba(220,38,38,0.22); color: {COLOR_DANGER}; }}

        /* --- Sidebar nav buttons --- */
        section[data-testid="stSidebar"] .stButton button {{
            border-radius: 10px;
            text-align: left;
            font-weight: 600;
        }}

        /* --- Confusion matrix mini cards --- */
        .cm-card {{
            border-radius: 12px;
            padding: 16px;
            text-align: center;
            font-weight: 700;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "best_fraud_model.joblib"
PREPROCESSOR_PATH = BASE_DIR / "models" / "preprocessor.joblib"

CLEANED_DATA_PATH = (
    BASE_DIR / "data" / "processed" / "cleaned_fraud_dataset.csv"
)

X_TEST_PATH = BASE_DIR / "data" / "processed" / "X_test.csv"
Y_TEST_PATH = BASE_DIR / "data" / "processed" / "y_test.csv"

REQUIRED_FILES = {
    "Trained model": MODEL_PATH,
    "Preprocessor": PREPROCESSOR_PATH,
    "Cleaned dataset": CLEANED_DATA_PATH,
    "Test features (X_test.csv)": X_TEST_PATH,
    "Test labels (y_test.csv)": Y_TEST_PATH,
}

# ============================================================
# FILE VALIDATION
# ============================================================

missing_files = [
    label for label, path in REQUIRED_FILES.items() if not path.exists()
]

if missing_files:
    st.error("The application could not start because required project files are missing.")
    st.markdown(
        "**Missing files:**\n" + "\n".join(f"- {name}" for name in missing_files)
    )
    st.stop()

# ============================================================
# LOAD FILES
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_preprocessor():
    return joblib.load(PREPROCESSOR_PATH)


@st.cache_data
def load_data():
    return pd.read_csv(CLEANED_DATA_PATH)


@st.cache_data
def load_test_data():
    X_test = pd.read_csv(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH).squeeze()
    return X_test, y_test


try:
    model = load_model()
    preprocessor = load_preprocessor()
    df = load_data()
    X_test, y_test = load_test_data()

except Exception as e:
    st.error("Unable to load project files.")
    with st.expander("Technical details"):
        st.code(str(e))
    st.stop()


# ============================================================
# FEATURE ENGINEERING  (unchanged from the original app.py)
# ============================================================

def create_features(transaction_df):
    """
    Reproduce the feature engineering used in Notebook 04.
    """

    data = transaction_df.copy()

    # Date conversion
    data["Date"] = pd.to_datetime(data["Date"])

    # --------------------------------------------------------
    # Temporal features
    # --------------------------------------------------------

    data["Year"] = data["Date"].dt.year
    data["Month"] = data["Date"].dt.month
    data["Day"] = data["Date"].dt.day
    data["Day_of_Week"] = data["Date"].dt.dayofweek

    # --------------------------------------------------------
    # Amount bin
    # --------------------------------------------------------

    amount_bins = [0, 10, 50, 100, 250, 500, np.inf]

    amount_labels = [
        "micro",
        "low",
        "medium",
        "high",
        "very_high",
        "extreme",
    ]

    data["Amount_Bin"] = pd.cut(
        data["Transaction_Amount"],
        bins=amount_bins,
        labels=amount_labels,
        include_lowest=True,
    )

    # --------------------------------------------------------
    # Balance-to-Amount Ratio
    # --------------------------------------------------------

    data["Balance_to_Amount_Ratio"] = np.where(
        data["Transaction_Amount"] == 0,
        np.nan,
        data["Account_Balance"] / data["Transaction_Amount"],
    )

    median_ratio = (
        data["Balance_to_Amount_Ratio"]
        .replace([np.inf, -np.inf], np.nan)
        .median()
    )

    data["Balance_to_Amount_Ratio"] = (
        data["Balance_to_Amount_Ratio"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(median_ratio)
    )

    # --------------------------------------------------------
    # User-level behavioral features
    # --------------------------------------------------------

    data["User_Transaction_Count"] = (
        data.groupby("User_ID")["Transaction_Amount"]
        .transform("count")
    )

    data["User_Avg_Transaction_Amount"] = (
        data.groupby("User_ID")["Transaction_Amount"]
        .transform("mean")
    )

    data["User_Total_Transaction_Amount"] = (
        data.groupby("User_ID")["Transaction_Amount"]
        .transform("sum")
    )

    data["User_Amount_Deviation"] = (
        data["Transaction_Amount"]
        - data["User_Avg_Transaction_Amount"]
    )

    return data


# ============================================================
# CREATE PREDICTION INPUT  (unchanged from the original app.py)
# ============================================================

def prepare_prediction(transaction):
    """
    Combine the new transaction with historical data so that
    user-level behavioral features are calculated in the same
    manner as the project's feature-engineering process.
    """

    transaction = transaction.copy()

    # We only need historical non-target data
    history = df.drop(columns=["Fraud_Label"], errors="ignore").copy()

    # Add the new transaction
    combined = pd.concat(
        [history, transaction],
        ignore_index=True
    )

    # Feature engineering
    combined = create_features(combined)

    # Last row is our new transaction
    prediction_row = combined.iloc[[-1]].copy()

    # --------------------------------------------------------
    # Features used by Notebook 05
    # --------------------------------------------------------

    numerical_features = [
        "Transaction_Amount",
        "Account_Balance",
        "Previous_Fraudulent_Activity",
        "Daily_Transaction_Count",
        "Card_Age",
        "Year",
        "Month",
        "Day",
        "Day_of_Week",
        "Balance_to_Amount_Ratio",
        "User_Transaction_Count",
        "User_Avg_Transaction_Amount",
        "User_Total_Transaction_Amount",
        "User_Amount_Deviation",
    ]

    categorical_features = [
        "Transaction_Type",
        "Device_Type",
        "Location",
        "Merchant_Category",
        "Card_Type",
        "Amount_Bin",
    ]

    all_features = numerical_features + categorical_features

    model_input = prediction_row[all_features]

    return model_input


def get_risk_level(probability):
    """Application-level interpretation of the model's fraud probability."""
    pct = probability * 100
    if pct < 30:
        return "Low Risk", "badge-low"
    elif pct < 60:
        return "Moderate Risk", "badge-moderate"
    elif pct < 80:
        return "High Risk", "badge-high"
    else:
        return "Very High Risk", "badge-veryhigh"


# ============================================================
# UI HELPER FUNCTIONS
# ============================================================

def render_section_header(title, subtitle=None, icon=""):
    st.markdown(
        f"""
        <div class="section-header">
            <h3>{icon} {title}</h3>
        </div>
        {f'<div class="section-sub">{subtitle}</div>' if subtitle else ''}
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(icon, label, value, subtext, accent_color):
    st.markdown(
        f"""
        <div class="kpi-card" style="border-top: 3px solid {accent_color};">
            <div class="kpi-icon">{icon}</div>
            <div class="kpi-label">{label}</div>
            <div class="kpi-value" style="color:{accent_color};">{value}</div>
            <div class="kpi-sub">{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_info_card(title, items, accent_color=COLOR_PRIMARY):
    items_html = "".join(f"<li>{item}</li>" for item in items)
    st.markdown(
        f"""
        <div class="info-card" style="border-left-color:{accent_color};">
            <h4>{title}</h4>
            <ul>{items_html}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_risk_card(is_fraud, probability):
    risk_label, risk_class = get_risk_level(probability)

    if is_fraud:
        st.markdown(
            f"""
            <div class="risk-card risk-card-fraud">
                <div class="risk-card-title">⚠️ Potential Fraud Detected</div>
                <div class="kpi-label">ESTIMATED FRAUD PROBABILITY</div>
                <div class="risk-card-prob" style="color:{COLOR_DANGER};">{probability * 100:.2f}%</div>
                <span class="badge {risk_class}">{risk_label}</span>
                <div class="risk-card-note" style="margin-top:12px;">
                    This is a model prediction, not a guarantee. Please review this
                    transaction carefully.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="risk-card risk-card-legit">
                <div class="risk-card-title">✓ Transaction Classified as Legitimate</div>
                <div class="kpi-label">ESTIMATED FRAUD PROBABILITY</div>
                <div class="risk-card-prob" style="color:{COLOR_SUCCESS};">{probability * 100:.2f}%</div>
                <span class="badge {risk_class}">{risk_label}</span>
                <div class="risk-card-note" style="margin-top:12px;">
                    The model currently estimates a lower fraud risk. This is an
                    estimated probability, not a guarantee of legitimacy.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.progress(min(float(probability), 1.0))


# ============================================================
# SIDEBAR
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

st.sidebar.markdown(
    f"""
    <div style="padding: 6px 0 14px 0;">
        <div style="font-size:20px; font-weight:800;">💳 Financial Fraud Detection</div>
        <div style="color:{COLOR_TEXT_MUTED}; font-size:13px; margin-top:4px;">
            Machine Learning powered transaction risk analysis
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

NAV_ITEMS = [
    ("Dashboard", "📊"),
    ("Fraud Prediction", "🔍"),
    ("Test Set Analysis", "📈"),
    ("Model Information", "🤖"),
]

for name, icon in NAV_ITEMS:
    is_active = st.session_state.page == name
    if st.sidebar.button(
        f"{icon}  {name}",
        use_container_width=True,
        type="primary" if is_active else "secondary",
        key=f"nav_{name}",
    ):
        st.session_state.page = name
        st.rerun()

st.sidebar.markdown("<hr style='border-color: rgba(148,163,184,0.15);'>", unsafe_allow_html=True)

st.sidebar.markdown(
    f"""
    <div style="font-size:13px; color:{COLOR_TEXT_MUTED}; line-height:1.6;">
        <b style="color:{COLOR_TEXT_LIGHT};">Model:</b> Logistic Regression<br>
        <b style="color:{COLOR_TEXT_LIGHT};">Purpose:</b> Detect potentially fraudulent
        financial transactions.<br><br>
        <b style="color:{COLOR_TEXT_LIGHT};">Classes:</b><br>
        🟢 0 → Legitimate<br>
        🔴 1 → Fraud
    </div>
    """,
    unsafe_allow_html=True,
)

page = st.session_state.page


# ============================================================
# PAGE 1 — DASHBOARD
# ============================================================

if page == "Dashboard":

    st.markdown(
        f"""
        <div style="margin-bottom: 4px;">
            <div style="font-size:34px; font-weight:900;">💳 Financial Fraud Detection</div>
            <div style="font-size:17px; color:{COLOR_TEXT_MUTED}; margin-top:2px;">
                Machine Learning &amp; Risk Analysis
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="app-caption" style="margin: 10px 0 22px 0; font-size:15px;">
            This dashboard analyzes real transaction data and applies a trained
            Logistic Regression model to estimate the likelihood that a
            financial transaction is fraudulent.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # Dataset metrics (calculated dynamically — never hard-coded)
    # --------------------------------------------------------

    total_transactions = len(df)
    fraud_count = int((df["Fraud_Label"] == 1).sum())
    legitimate_count = int((df["Fraud_Label"] == 0).sum())
    fraud_rate = fraud_count / total_transactions * 100

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        render_metric_card("📦", "Total Transactions", f"{total_transactions:,}",
                            "Transactions analyzed", COLOR_PRIMARY)
    with c2:
        render_metric_card("🚨", "Fraud Transactions", f"{fraud_count:,}",
                            "Potentially fraudulent", COLOR_DANGER)
    with c3:
        render_metric_card("✅", "Legitimate Transactions", f"{legitimate_count:,}",
                            "Normal transactions", COLOR_SUCCESS)
    with c4:
        render_metric_card("📊", "Fraud Rate", f"{fraud_rate:.2f}%",
                            "Share of total transactions", COLOR_WARNING)

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # Class distribution chart
    # --------------------------------------------------------

    render_section_header("Transaction Class Distribution", icon="🧭")

    chart_col, text_col = st.columns([2, 1])

    class_data = pd.DataFrame(
        {
            "Class": ["Legitimate", "Fraud"],
            "Transactions": [legitimate_count, fraud_count],
        }
    )
    with chart_col:
        # Dynamic Y-axis maximum with enough headroom above the tallest bar
        y_max = max(legitimate_count, fraud_count) * 1.15

        chart = (
            alt.Chart(class_data)
            .mark_bar(
                cornerRadiusTopLeft=6,
                cornerRadiusTopRight=6,
                size=120,
            )
            .encode(
                # -----------------------------
                # X-AXIS
                # -----------------------------
                x=alt.X(
                    "Class:N",
                    sort=["Legitimate", "Fraud"],
                    title="Transaction Class",
                    axis=alt.Axis(
                        labelAngle=0,
                        labelFontSize=12,
                        labelColor=COLOR_TEXT_MUTED,
                        titleColor=COLOR_TEXT_LIGHT,
                        titleFontSize=13,
                        titlePadding=14,
                        labelPadding=8,
                        domain=True,
                        ticks=True,
                    ),
                ),

                # -----------------------------
                # Y-AXIS
                # -----------------------------
                y=alt.Y(
                    "Transactions:Q",
                    title="Transactions",
                    scale=alt.Scale(
                        domain=[0, y_max]
                    ),
                    axis=alt.Axis(
                        labelColor=COLOR_TEXT_MUTED,
                        titleColor=COLOR_TEXT_LIGHT,
                        titleFontSize=13,
                        titlePadding=12,
                        labelFontSize=11,
                        domain=True,
                        ticks=True,
                        grid=True,
                        gridColor="rgba(148,163,184,0.15)",
                        gridDash=[3, 3],
                    ),
                ),

                # -----------------------------
                # BAR COLORS
                # -----------------------------
                color=alt.Color(
                    "Class:N",
                    scale=alt.Scale(
                        domain=["Legitimate", "Fraud"],
                        range=[
                            COLOR_SUCCESS,
                            COLOR_DANGER,
                        ],
                    ),
                    legend=None,
                ),

                # -----------------------------
                # HOVER TOOLTIP
                # -----------------------------
                tooltip=[
                    alt.Tooltip(
                        "Class:N",
                        title="Transaction Class",
                    ),
                    alt.Tooltip(
                        "Transactions:Q",
                        title="Transactions",
                        format=",",
                    ),
                ],
            )
            .properties(
                height=300,
                padding={
                    "top": 20,
                    "bottom": 5,
                    "left": 10,
                    "right": 10,
                },
            )
            .configure_view(
                strokeWidth=0,
            )
        )

        st.altair_chart(
            chart,
            use_container_width=True,
        )
        
    with text_col:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="info-card">
                <h4>What this shows</h4>
                <div style="color:{COLOR_TEXT_MUTED}; font-size:14px; line-height:1.6;">
                    Most transactions in the dataset are legitimate, while
                    approximately <b style="color:{COLOR_TEXT_LIGHT};">{fraud_rate:.2f}%</b>
                    are labeled as fraudulent.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # Dataset Insights
    # --------------------------------------------------------

    render_section_header("Dataset Insights", icon="🔎")

    avg_amount = df["Transaction_Amount"].mean()
    max_amount = df["Transaction_Amount"].max()
    avg_balance = df["Account_Balance"].mean()
    unique_users = df["User_ID"].nunique()

    i1, i2, i3, i4 = st.columns(4)

    with i1:
        render_metric_card("💵", "Average Transaction", f"{avg_amount:,.2f}",
                            "Mean transaction amount", COLOR_PRIMARY)
    with i2:
        render_metric_card("🏦", "Average Account Balance", f"{avg_balance:,.2f}",
                            "Mean balance across users", COLOR_SECONDARY)
    with i3:
        render_metric_card("📈", "Highest Transaction", f"{max_amount:,.2f}",
                            "Largest single transaction", COLOR_WARNING)
    with i4:
        render_metric_card("👥", "Unique Users", f"{unique_users:,}",
                            "Distinct users in the dataset", COLOR_SUCCESS)

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # Dataset preview
    # --------------------------------------------------------

    render_section_header("Dataset Preview", icon="📋")

    display_columns = [
        "Transaction_ID",
        "User_ID",
        "Transaction_Amount",
        "Transaction_Type",
        "Device_Type",
        "Location",
        "Merchant_Category",
        "Card_Type",
        "Fraud_Label",
    ]

    available_columns = [c for c in display_columns if c in df.columns]

    n_rows = st.slider("Rows to display", min_value=5, max_value=50, value=10, step=5)

    st.dataframe(
        df[available_columns].head(n_rows),
        use_container_width=True,
    )
    st.caption("Preview of the transactions used by the application.")


# ============================================================
# PAGE 2 — FRAUD PREDICTION
# ============================================================

elif page == "Fraud Prediction":

    st.markdown(
        f"""
        <div style="font-size:30px; font-weight:900;">🔍 Fraud Risk Prediction</div>
        <div style="color:{COLOR_TEXT_MUTED}; font-size:15px; margin-top:2px; margin-bottom:20px;">
            Enter transaction details to estimate potential fraud risk.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # Section 1 — Transaction Information
    # --------------------------------------------------------

    render_section_header("Transaction Information", icon="🧾")

    t1, t2, t3 = st.columns(3)

    with t1:
        user_id = st.text_input("User ID", value=str(df["User_ID"].iloc[0]))
        transaction_amount = st.number_input(
            "Transaction Amount", min_value=0.0, max_value=1000000.0,
            value=100.0, step=10.0,
        )

    with t2:
        account_balance = st.number_input(
            "Account Balance", min_value=0.0, max_value=10000000.0,
            value=5000.0, step=100.0,
        )
        transaction_type = st.selectbox(
            "Transaction Type", sorted(df["Transaction_Type"].dropna().unique()),
        )

    with t3:
        transaction_date = st.date_input(
            "Transaction Date", value=pd.Timestamp("2023-06-01").date(),
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # Section 2 — Device & Location
    # --------------------------------------------------------

    render_section_header("Device & Location", icon="📍")

    d1, d2, d3, d4 = st.columns(4)

    with d1:
        device_type = st.selectbox("Device Type", sorted(df["Device_Type"].dropna().unique()))
    with d2:
        location = st.selectbox("Location", sorted(df["Location"].dropna().unique()))
    with d3:
        merchant_category = st.selectbox(
            "Merchant Category", sorted(df["Merchant_Category"].dropna().unique())
        )
    with d4:
        card_type = st.selectbox("Card Type", sorted(df["Card_Type"].dropna().unique()))

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # Section 3 — Risk Indicators
    # --------------------------------------------------------

    render_section_header("Risk Indicators", icon="⚙️")

    r1, r2, r3 = st.columns(3)

    with r1:
        previous_fraud = st.selectbox(
            "Previous Fraudulent Activity", [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
        )
    with r2:
        daily_transaction_count = st.number_input(
            "Daily Transaction Count", min_value=1, max_value=100, value=5, step=1,
        )
    with r3:
        card_age = st.number_input(
            "Card Age (days)", min_value=1, max_value=5000, value=100, step=1,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    predict_clicked = st.button(
        "🚨 Check Fraud Risk", use_container_width=True, type="primary"
    )

    if predict_clicked:

        transaction = pd.DataFrame(
            [
                {
                    "Transaction_ID": "NEW_TRANSACTION",
                    "User_ID": str(user_id),
                    "Transaction_Amount": transaction_amount,
                    "Transaction_Type": transaction_type,
                    "Date": str(transaction_date),
                    "Account_Balance": account_balance,
                    "Device_Type": device_type,
                    "Location": location,
                    "Merchant_Category": merchant_category,
                    "Previous_Fraudulent_Activity": previous_fraud,
                    "Daily_Transaction_Count": daily_transaction_count,
                    "Card_Type": card_type,
                    "Card_Age": card_age,
                }
            ]
        )

        try:
            # Create the exact feature structure
            model_input = prepare_prediction(transaction)

            # Transform using saved preprocessing pipeline
            transformed_input = preprocessor.transform(model_input)

            # Prediction
            prediction = model.predict(transformed_input)[0]

            # Probability
            probability = model.predict_proba(transformed_input)[0][1]

            st.markdown("<br>", unsafe_allow_html=True)
            render_risk_card(is_fraud=(prediction == 1), probability=probability)

        except Exception as e:
            st.error("Prediction could not be completed.")
            with st.expander("Technical details"):
                st.code(str(e))


# ============================================================
# PAGE 3 — TEST SET ANALYSIS
# ============================================================

elif page == "Test Set Analysis":

    st.markdown(
        f"""
        <div style="font-size:30px; font-weight:900;">📈 Model Performance</div>
        <div style="color:{COLOR_TEXT_MUTED}; font-size:15px; margin-top:2px; margin-bottom:20px;">
            Evaluation on the held-out test dataset
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Generate predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics (calculated dynamically from X_test / y_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)

    m1, m2, m3, m4, m5 = st.columns(5)

    with m1:
        render_metric_card("🎯", "Accuracy", f"{accuracy:.4f}", "Overall correctness", COLOR_PRIMARY)
    with m2:
        render_metric_card("🔬", "Precision", f"{precision:.4f}", "Of predicted fraud", COLOR_SECONDARY)
    with m3:
        render_metric_card("📡", "Recall", f"{recall:.4f}", "Of actual fraud caught", COLOR_WARNING)
    with m4:
        render_metric_card("⚖️", "F1 Score", f"{f1:.4f}", "Precision/recall balance", COLOR_SUCCESS)
    with m5:
        render_metric_card("📉", "ROC-AUC", f"{roc_auc:.4f}", "Discrimination ability", COLOR_DANGER)

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    render_section_header("Confusion Matrix", icon="🧮")
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    # Two equal-width columns
    cm_col, legend_col = st.columns([1, 1], gap="medium")
    # --------------------------------------------------------
    # Left: Confusion Matrix Table
    # --------------------------------------------------------

    with cm_col:

        cm_df = pd.DataFrame(
            cm,
            index=[
                "Actual Legitimate",
                "Actual Fraud",
            ],
            columns=[
                "Predicted Legitimate",
                "Predicted Fraud",
            ],
        )

        st.dataframe(
            cm_df,
            width="stretch",
            height="auto",
            hide_index=False,
        )

    # --------------------------------------------------------
    # Right: Reading the Matrix
    # --------------------------------------------------------

    with legend_col:
        render_info_card(
            "Reading the matrix",
            [
                "Rows represent the actual class",
                "Columns represent the predicted class",
                "Diagonal cells are correct predictions",
            ],
        )
        
    st.markdown("<br>", unsafe_allow_html=True)
    render_section_header("Prediction Breakdown", icon="🧩")

    b1, b2, b3, b4 = st.columns(4)

    def render_cm_card(col, label, value, color):
        col.markdown(
            f"""
            <div class="cm-card" style="background: rgba({color}, 0.15); color: rgb({color});">
                <div style="font-size:13px; font-weight:700; letter-spacing:0.03em;">{label}</div>
                <div style="font-size:26px; font-weight:900; margin-top:4px;">{value:,}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    render_cm_card(b1, "TRUE NEGATIVES", tn, "22,163,74")
    render_cm_card(b2, "FALSE POSITIVES", fp, "245,158,11")
    render_cm_card(b3, "FALSE NEGATIVES", fn, "220,38,38")
    render_cm_card(b4, "TRUE POSITIVES", tp, "37,99,235")

    st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# PAGE 4 — MODEL INFORMATION
# ============================================================

elif page == "Model Information":

    st.markdown(
        f"""
        <div style="font-size:30px; font-weight:900;">
            🤖 Model Information
        </div>
        <div style="
            color:{COLOR_TEXT_MUTED};
            font-size:15px;
            margin-top:2px;
            margin-bottom:20px;
        ">
            How the model was selected, trained, and evaluated
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # Selected Model
    # --------------------------------------------------------

    render_section_header("Selected Model", icon="🏆")
    with st.container(border=True):

        st.markdown("### Logistic Regression")

        st.write(
            "The project trained and compared three classification models — "
            "Logistic Regression, Random Forest, and Gradient Boosting. "
            "Logistic Regression was selected because it achieved the "
            "highest F1-Score on the held-out test set."
        )
   
    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # Model Selection Results
    # --------------------------------------------------------

    render_section_header("Model Selection Results", icon="📊")

    comparison = pd.DataFrame(
        {
            "Model": [
                "Logistic Regression",
                "Gradient Boosting",
                "Random Forest",
            ],
            "F1-Score": [
                0.3823,
                0.0025,
                0.0000,
            ],
            "ROC-AUC": [
                0.4922,
                0.4924,
                0.5023,
            ],
        }
    )
    # Center F1-Score and ROC-AUC values
    st.markdown(
        """
        <style>
        div[data-testid="stTable"] table th:nth-child(2),
        div[data-testid="stTable"] table th:nth-child(3) {
            text-align: center !important;
        }

        div[data-testid="stTable"] table td:nth-child(2),
        div[data-testid="stTable"] table td:nth-child(3) {
            text-align: center !important;
        }

        div[data-testid="stTable"] table th:first-child,
        div[data-testid="stTable"] table td:first-child {
            text-align: left !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.table(
        comparison.style.format(
            {
                "F1-Score": "{:.4f}",
                "ROC-AUC": "{:.4f}",
            }
        )
    )

    st.markdown("<br>", unsafe_allow_html=True)
    
    

    # --------------------------------------------------------
    # Preprocessing Pipeline
    # --------------------------------------------------------

    render_section_header("Preprocessing Pipeline", icon="🛠️")

    # Feature Engineering
    with st.container(border=True):

        st.markdown("### 🔧 Feature Engineering")

        fe1, fe2, fe3 = st.columns(3, gap="medium")

        with fe1:
            st.markdown("**📅 Date Features**")
            st.caption(
                "Date → Year, Month, Day, Day_of_Week"
            )

        with fe2:
            st.markdown("**💰 Transaction Features**")
            st.caption(
                "Amount Binning + Balance-to-Amount Ratio"
            )

        with fe3:
            st.markdown("**👤 User Behaviour Features**")
            st.caption(
                "Transaction Count, Average, Total & Amount Deviation"
            )

    # Arrow between feature engineering and preprocessing
    st.markdown(
        """
        <div style="
            text-align:center;
            font-size:28px;
            font-weight:800;
            margin:10px 0;
        ">
            ↓
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Numerical and Categorical Processing
    pp1, pp2 = st.columns(2, gap="medium")

    with pp1:

        with st.container(border=True):

            st.markdown("### 🔢 Numerical Features")

            st.markdown("**① Median Imputation**")

            st.markdown(
                "<div style='font-size:22px; margin:8px 0;'>↓</div>",
                unsafe_allow_html=True,
            )

            st.markdown("**② StandardScaler**")

    with pp2:

        with st.container(border=True):

            st.markdown("### 🏷️ Categorical Features")

            st.markdown("**① Most-Frequent Imputation**")

            st.markdown(
                "<div style='font-size:22px; margin:8px 0;'>↓</div>",
                unsafe_allow_html=True,
            )

            st.markdown("**② One-Hot Encoding**")

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # Model Features
    # --------------------------------------------------------

    render_section_header("Model Features", icon="🧬")

    numerical_features = [
        "Transaction_Amount",
        "Account_Balance",
        "Previous_Fraudulent_Activity",
        "Daily_Transaction_Count",
        "Card_Age",
        "Year",
        "Month",
        "Day",
        "Day_of_Week",
        "Balance_to_Amount_Ratio",
        "User_Transaction_Count",
        "User_Avg_Transaction_Amount",
        "User_Total_Transaction_Amount",
        "User_Amount_Deviation",
    ]

    categorical_features = [
        "Transaction_Type",
        "Device_Type",
        "Location",
        "Merchant_Category",
        "Card_Type",
        "Amount_Bin",
    ]

    f1_col, f2_col = st.columns(2, gap="medium")

    with f1_col:

        with st.expander(
            f"🔢 Numerical Features ({len(numerical_features)}) — click to explore",
            expanded=False,
        ):

            for i, feature in enumerate(
                numerical_features,
                start=1,
            ):
                st.markdown(
                    f"**{i}.** `{feature.replace('_', ' ')}`"
                )

    with f2_col:

        with st.expander(
            f"🏷️ Categorical Features ({len(categorical_features)}) — click to explore",
            expanded=False,
        ):

            for i, feature in enumerate(
                categorical_features,
                start=1,
            ):
                st.markdown(
                    f"**{i}.** `{feature.replace('_', ' ')}`"
                )

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # Footer
    # --------------------------------------------------------

    st.markdown(
        f"""
        <div class="app-caption"
             style="
                 text-align:center;
                 font-size:13px;
                 margin-top:20px;
             ">
            Financial Fraud Detection — Machine Learning &amp; Streamlit
        </div>
        """,
        unsafe_allow_html=True,
    )