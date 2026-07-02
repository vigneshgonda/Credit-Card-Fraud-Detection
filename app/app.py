import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_fraud_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"
REPORT_PATH = BASE_DIR / "outputs" / "reports" / "model_comparison.csv"
FIGURES_DIR = BASE_DIR / "outputs" / "figures"

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed"
)

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

model, scaler = load_artifacts()

st.markdown("""
<style>
.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

.hero {
    background: linear-gradient(135deg, #0f3d5e 0%, #1f77b4 100%);
    padding: 36px 38px;
    border-radius: 22px;
    color: white;
    margin-bottom: 28px;
    box-shadow: 0 10px 30px rgba(15, 61, 94, 0.20);
}
.hero-title {
    font-size: 42px;
    font-weight: 900;
    margin-bottom: 8px;
}
.hero-subtitle {
    font-size: 18px;
    opacity: 0.95;
}
.kpi-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 18px;
    padding: 24px;
    text-align: center;
    box-shadow: 0 6px 18px rgba(15, 61, 94, 0.08);
    min-height: 130px;
}
.kpi-label {
    color: #64748b;
    font-size: 14px;
    font-weight: 700;
    margin-bottom: 10px;
}
.kpi-value {
    color: #0f3d5e;
    font-size: 30px;
    font-weight: 900;
}
.card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 18px;
    padding: 24px;
    box-shadow: 0 6px 18px rgba(15, 61, 94, 0.06);
    margin-bottom: 18px;
}
.card-title {
    font-size: 22px;
    font-weight: 800;
    color: #0f3d5e;
    margin-bottom: 10px;
}
.card-text {
    font-size: 16px;
    color: #334155;
    line-height: 1.6;
}
.success-box {
    background: #dcfce7;
    color: #166534;
    border: 1px solid #86efac;
    padding: 22px;
    border-radius: 18px;
    font-size: 24px;
    font-weight: 900;
    text-align: center;
}
.danger-box {
    background: #fee2e2;
    color: #991b1b;
    border: 1px solid #fca5a5;
    padding: 22px;
    border-radius: 18px;
    font-size: 24px;
    font-weight: 900;
    text-align: center;
}
.note-box {
    background: #eff6ff;
    color: #1e3a8a;
    border: 1px solid #bfdbfe;
    padding: 18px;
    border-radius: 16px;
    line-height: 1.5;
}
.section-title {
    font-size: 28px;
    font-weight: 900;
    color: #0f3d5e;
    margin-top: 26px;
    margin-bottom: 14px;
}
.footer {
    text-align: center;
    color: #64748b;
    margin-top: 35px;
    font-size: 14px;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 12px;
    padding: 10px 18px;
    background-color: #f1f5f9;
}
.stTabs [aria-selected="true"] {
    background-color: #dbeafe;
    color: #0f3d5e;
    font-weight: 800;
}
@media (max-width: 900px) {
    .hero-title {font-size: 30px;}
    .hero-subtitle {font-size: 15px;}
    .kpi-value {font-size: 24px;}
}
</style>
""", unsafe_allow_html=True)

def hero():
    st.markdown("""
    <div class="hero">
        <div class="hero-title">💳 Credit Card Fraud Detection</div>
        <div class="hero-subtitle">
            Professional machine learning dashboard for detecting fraudulent credit card transactions.
        </div>
    </div>
    """, unsafe_allow_html=True)

def kpi_card(label, value):
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)

def section(title):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)

def info_card(title, text):
    st.markdown(f"""
    <div class="card">
        <div class="card-title">{title}</div>
        <div class="card-text">{text}</div>
    </div>
    """, unsafe_allow_html=True)

def show_image(title, filename):
    path = FIGURES_DIR / filename
    st.markdown(f"### {title}")
    if path.exists():
        st.image(str(path), use_container_width=True)
    else:
        st.warning(f"{filename} not found.")

def load_report():
    if REPORT_PATH.exists():
        return pd.read_csv(REPORT_PATH)
    return None

hero()

tab_home, tab_predict, tab_perf, tab_viz, tab_about = st.tabs(
    ["🏠 Home", "🔍 Predict", "📊 Model Performance", "📈 Visualizations", "ℹ️ About"]
)

with tab_home:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        kpi_card("Total Transactions", "284,807")
    with col2:
        kpi_card("Fraud Cases", "492")
    with col3:
        kpi_card("Best Model", "Random Forest")
    with col4:
        kpi_card("F1 Score", "99.98%")

    section("📌 Project Overview")
    info_card(
        "Problem Statement",
        "Credit card fraud is rare but costly. This project builds a machine learning system to classify transactions as normal or fraudulent using the popular credit card fraud dataset."
    )

    info_card(
        "Why SMOTE?",
        "The dataset is highly imbalanced: only 492 fraud cases exist among 284,807 transactions. SMOTE was used to create synthetic fraud examples and help the model learn fraud patterns better."
    )

    section("🔁 Workflow")
    st.markdown("""
    <div class="card">
        <div class="card-text">
            <b>Dataset</b> → <b>EDA</b> → <b>SMOTE Balancing</b> → <b>Feature Scaling</b> →
            <b>Model Training</b> → <b>Evaluation</b> → <b>Dashboard</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

    section("📸 Quick Preview")
    col1, col2 = st.columns(2)
    with col1:
        show_image("Confusion Matrix", "confusion_matrix.png")
    with col2:
        show_image("Feature Importance", "feature_importance.png")

with tab_predict:
    section("🔍 Transaction Fraud Prediction")

    st.markdown("""
    <div class="note-box">
        <b>Important:</b> V1 to V28 are PCA-transformed confidential features.
        For demo usage, you can keep them as 0.0, or expand the custom input section and enter values manually.
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    col1, col2, col3 = st.columns(3)
    with col1:
        time = st.number_input("Transaction Time", min_value=0.0, value=0.0)
    with col2:
        amount = st.number_input("Transaction Amount", min_value=0.0, value=100.0)
    with col3:
        use_custom = st.checkbox("Enter custom V1-V28 values")

    sample = {"Time": time}
    if use_custom:
        st.markdown("#### PCA Feature Inputs")
        input_cols = st.columns(4)
        for i in range(1, 29):
            with input_cols[(i - 1) % 4]:
                sample[f"V{i}"] = st.number_input(f"V{i}", value=0.0, key=f"v{i}")
    else:
        for i in range(1, 29):
            sample[f"V{i}"] = 0.0

    sample["Amount"] = amount
    columns = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    input_df = pd.DataFrame([sample])[columns]

    with st.expander("Preview transaction input"):
        st.dataframe(input_df, use_container_width=True)

    if st.button("🚀 Predict Transaction", use_container_width=True):
        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)[0]
        probability = model.predict_proba(scaled_input)[0][1]

        st.markdown("### Prediction Result")
        if prediction == 1:
            st.markdown('<div class="danger-box">🚨 Fraud Transaction Detected</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="success-box">✅ Normal Transaction</div>', unsafe_allow_html=True)

        st.markdown("#### Fraud Probability")
        st.progress(float(probability))
        st.write(f"Fraud probability: **{probability:.4f}**")

with tab_perf:
    section("📊 Model Performance")

    report_df = load_report()
    if report_df is not None:
        st.dataframe(report_df, use_container_width=True)
        best_row = report_df.sort_values("F1 Score", ascending=False).iloc[0]

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            kpi_card("Best Model", best_row["Model"])
        with col2:
            kpi_card("Accuracy", f"{best_row['Accuracy'] * 100:.2f}%")
        with col3:
            kpi_card("Recall", f"{best_row['Recall'] * 100:.2f}%")
        with col4:
            kpi_card("F1 Score", f"{best_row['F1 Score'] * 100:.2f}%")

        csv = report_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Model Comparison CSV",
            csv,
            "model_comparison.csv",
            "text/csv",
            use_container_width=True
        )

        info_card(
            "Metric Interpretation",
            "Recall is very important in fraud detection because missing a fraud transaction can be costly. Precision helps reduce false alarms. F1-score balances both precision and recall."
        )
    else:
        st.warning("Model comparison report not found.")

with tab_viz:
    section("📈 Visualizations")

    viz_tab1, viz_tab2, viz_tab3 = st.tabs(
        ["Dataset Analysis", "Model Evaluation", "Feature Insights"]
    )

    with viz_tab1:
        col1, col2 = st.columns(2)
        with col1:
            show_image("Class Distribution", "class_distribution.png")
        with col2:
            show_image("Amount Distribution", "amount_distribution.png")

    with viz_tab2:
        col1, col2 = st.columns(2)
        with col1:
            show_image("Confusion Matrix", "confusion_matrix.png")
        with col2:
            show_image("ROC Curve", "roc_curve.png")
        show_image("Precision-Recall Curve", "precision_recall_curve.png")

    with viz_tab3:
        show_image("Top 10 Feature Importance", "feature_importance.png")

with tab_about:
    section("ℹ️ About This Project")

    info_card(
        "Project Summary",
        "This project demonstrates an end-to-end machine learning workflow for fraud detection. It includes data loading, EDA, SMOTE balancing, model comparison, Random Forest selection, evaluation visualizations, and a Streamlit dashboard."
    )

    info_card(
        "Technologies Used",
        "Python, Pandas, NumPy, Scikit-learn, Imbalanced-learn, Matplotlib, Seaborn, Streamlit, Joblib."
    )

    info_card(
        "Dataset",
        "The dataset contains anonymized credit card transactions. Features V1 to V28 are PCA-transformed for privacy. The target variable Class contains 0 for normal transactions and 1 for fraud."
    )

    info_card(
        "Future Improvements",
        "Add XGBoost or LightGBM, SHAP explainability, REST API deployment, real-time fraud monitoring simulation, and more realistic transaction input samples."
    )

st.markdown(
    '<div class="footer">Built with Python, Scikit-learn, SMOTE, Random Forest and Streamlit</div>',
    unsafe_allow_html=True
)

