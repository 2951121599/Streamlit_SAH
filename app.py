import streamlit as st
import pandas as pd
import numpy as np
import shap
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Configure matplotlib for better visualization
rcParams['font.family'] = 'sans-serif'
rcParams['font.size'] = 12

# Set page config with custom icon
st.set_page_config(
    page_title="Epilepsy Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    .stButton>button {background-color: #007bff; color: white; border-radius: 8px;}
    .stNumberInput>label {font-weight: bold; color: #2c3e50;}
    .sidebar .sidebar-content {background-color: #e9ecef;}
    h1 {color: #2c3e50; text-align: center;}
    h2 {color: #34495e; border-bottom: 2px solid #17a2b8; padding-bottom: 5px;}
    </style>
""", unsafe_allow_html=True)

# Title and introduction
st.title("🧠 Epilepsy Risk Prediction")
st.markdown("""
    This tool uses transcriptome data to predict Epilepsy risk and provides mechanistic insights 
    using SHAP (SHapley Additive exPlanations) visualizations. Adjust gene expression levels in the sidebar 
    to explore their impact on neuronal excitability and seizure susceptibility.
""")

# Load and prepare background data
@st.cache_data
def load_background_data():
    df = pd.read_excel('data/Epilepsy_data.xlsx')  # 更新为癫痫数据文件
    return df[['STXBP1', 'KCNQ2', 'CDKL5', 'GRIN2A', 'DEPDC5',
              'GABRG2', 'SLC2A1', 'LGI1', 'TSC2', 'ARX']]

# Load the pre-trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('data/Epilepsy_MODEL.h5')  # 更新为癫痫模型

# Initialize data and model
background_data = load_background_data()
model = load_model()

# Default values for genes
default_values = {
    'STXBP1': 11.1724, 'KCNQ2': 7.85244, 'CDKL5': 8.42552,
    'GRIN2A': 9.02267, 'DEPDC5': 6.97955, 'GABRG2': 9.65622,
    'SLC2A1': 8.81985, 'LGI1': 7.56641, 'TSC2': 8.21817, 'ARX': 7.01142
}

# Sidebar configuration
st.sidebar.header("🧬 Gene Expression Inputs")
st.sidebar.markdown("Adjust expression levels of epilepsy-related genes:")

# Reset button
if st.sidebar.button("Reset to Defaults", key="reset"):
    st.session_state.update(default_values)

# Dynamic two-column layout for 10 genes
gene_features = list(default_values.keys())
gene_values = {}
cols = st.sidebar.columns(2)  # 改为两列布局

for i, gene in enumerate(gene_features):
    with cols[i % 2]:  # 按两列分布
        gene_values[gene] = st.number_input(
            gene,
            min_value=float(background_data[gene].min()),
            max_value=float(background_data[gene].max()),
            value=default_values[gene],
            step=0.01,
            format="%.2f",
            key=gene
        )

# Prepare input data
def prepare_input_data():
    return pd.DataFrame([gene_values])

# Main analysis
if st.button("🔬 Analyze Gene Impacts", key="calculate"):
    input_df = prepare_input_data()
    
    # Prediction
    prediction = model.predict(input_df.values, verbose=0)[0][0]
    st.header("📈 Risk Prediction")
    st.metric("Epilepsy Risk Score", f"{prediction:.4f}", 
             delta="High Risk" if prediction >= 0.5 else "Low Risk",
             delta_color="inverse")
    
    # SHAP explanation
    explainer = shap.DeepExplainer(model, background_data.values)
    shap_values = np.squeeze(np.array(explainer.shap_values(input_df.values)))
    base_value = float(explainer.expected_value[0].numpy())
    
    # Visualization tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Force Plot", "Waterfall Plot", "Decision Plot", "Mechanistic Insights"])
    
    with tab1:
        st.subheader("Feature Impact Visualization")
        explanation = shap.Explanation(
            values=shap_values, 
            base_values=base_value, 
            feature_names=input_df.columns,
            data=input_df.values
        )
        shap.plots.force(explanation, matplotlib=True, show=False, figsize=(20, 4))
        st.pyplot(plt.gcf(), clear_figure=True)
    
    with tab4:
        st.subheader("Mechanistic Insights")
        st.markdown("""
        **Key Epilepsy-related Pathways:**
        - STXBP1: 突触囊泡对接和神经递质释放
        - KCNQ2: 神经元膜电位调节（M电流）
        - CDKL5: 神经发育和树突形成
        - GRIN2A: NMDA受体介导的突触可塑性
        - SLC2A1: 脑能量代谢（葡萄糖转运）
        """)
        importance_df = pd.DataFrame({'Gene': input_df.columns, 'SHAP Value': shap_values})
        importance_df = importance_df.sort_values('SHAP Value', ascending=False)
        st.dataframe(importance_df.style.background_gradient(cmap='coolwarm', subset=['SHAP Value']))

# Update documentation
with st.expander("📚 About This Epilepsy Analysis", expanded=True):
    st.markdown("""
    ### Model Overview
    本深度学习模型分析10个关键癫痫相关基因，涉及：
    - 离子通道功能
    - 突触传递调控
    - 神经发育过程
    - 脑能量代谢
    
    ### SHAP解释指南
    1. **力导向图 (Force Plot)**：显示各基因对风险评分的推拉效应
    2. **瀑布图 (Waterfall Plot)**：特征贡献的逐步可视化
    3. **决策图 (Decision Plot)**：累积效应可视化
    4. **机制解析 (Mechanistic Insights)**：结合SHAP值和已知生物学机制的分析
    """)

# Footer
st.markdown("---")
st.markdown(f"Developed for Epilepsy Research | Updated: {pd.Timestamp.now().strftime('%Y-%m-%d')}")