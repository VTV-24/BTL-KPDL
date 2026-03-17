import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt

# ===== CONFIG =====
st.set_page_config(
    page_title="Customer Analytics Dashboard",
    layout="centered"
)

# ===== CSS =====
st.markdown("""
<style>
img {
    max-width: 100% !important;
    height: auto !important;
}
.block-container {
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

st.title("📊 Customer Analytics Dashboard")

# ===== LOAD DATA =====
@st.cache_data
def load_data():
    base = "D:/KPDL/BTL/outputs"

    data = {}
    data["top_products"] = pd.read_csv(f"{base}/tables/top_products.csv")
    data["top_rules"] = pd.read_csv(f"{base}/tables/top_rules.csv")
    data["cluster_stats"] = pd.read_csv(f"{base}/tables/cluster_stats.csv")
    data["model_metrics"] = pd.read_csv(f"{base}/tables/model_metrics.csv")
    data["fig_path"] = f"{base}/figures"

    return data

data = load_data()

# ===== KPI =====
st.subheader("📌 Tổng quan nhanh")

col1, col2, col3, col4 = st.columns(4)

col1.metric("👥 Customers", len(data["cluster_stats"]))
col2.metric("📦 Products", len(data["top_products"]))
col3.metric("🧩 Clusters", data["cluster_stats"]["Cluster"].nunique())

best_model = data["model_metrics"].sort_values("accuracy", ascending=False).iloc[0]
col4.metric("🏆 Best Model", best_model["model"], f"{best_model['accuracy']:.2%}")

st.markdown("---")

# ===== TABS =====
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Tổng quan",
    "🛒 Association",
    "👥 Segmentation",
    "🤖 Modeling"
])

# ================= TAB 1 =================
with tab1:
    st.header("📊 Top sản phẩm")

    top_n = st.slider("Chọn số sản phẩm", 5, 20, 10)

    df = data["top_products"].head(top_n)
    st.dataframe(df, use_container_width=True)

    fig, ax = plt.subplots(figsize=(6,3))
    df.plot(
        x=df.columns[0],
        y=df.columns[1],
        kind="bar",
        ax=ax
    )
    st.pyplot(fig)

# ================= TAB 2 =================
with tab2:
    st.header("🛒 Association Rules")

    min_lift = st.slider("Lọc theo Lift >=", 1.0, 5.0, 1.0)

    df = data["top_rules"]
    df = df[df["lift"] >= min_lift]

    st.dataframe(df.head(10), use_container_width=True)

# ================= TAB 3 =================
with tab3:
    st.header("👥 Customer Segmentation")

    cluster = st.selectbox(
        "Chọn cluster",
        sorted(data["cluster_stats"]["Cluster"].unique())
    )

    df = data["cluster_stats"]
    st.dataframe(df[df["Cluster"] == cluster], use_container_width=True)

    st.subheader("Phân bố cluster")

    fig, ax = plt.subplots(figsize=(6,3))
    ax.bar(df["Cluster"], df["Count"])
    st.pyplot(fig)

# ================= TAB 4 =================
with tab4:
    st.header("🤖 Modeling")

    metric = st.selectbox(
        "Chọn metric",
        ["accuracy", "precision", "recall", "f1"]
    )

    fig, ax = plt.subplots(figsize=(6,3))
    data["model_metrics"].set_index("model")[metric].plot(kind="bar", ax=ax)
    st.pyplot(fig)

    st.dataframe(data["model_metrics"], use_container_width=True)

    # ===== ẢNH =====
    st.subheader("📌 Visualization")

    cm_path = os.path.join(data["fig_path"], "confusion_matrix.png")
    fi_path = os.path.join(data["fig_path"], "feature_importance.png")

    col_left, col_mid, col_right = st.columns([1,2,1])

    with col_mid:
        col1, col2 = st.columns(2)

        with col1:
            st.caption("Confusion Matrix")
            if os.path.exists(cm_path):
                st.image(cm_path, use_column_width=True)

        with col2:
            st.caption("Feature Importance")
            if os.path.exists(fi_path):
                st.image(fi_path, use_column_width=True)

# ===== FOOTER =====
st.markdown("---")
st.markdown("🚀 Project KPDL - Customer Analytics Dashboard")