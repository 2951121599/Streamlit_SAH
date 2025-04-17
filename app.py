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
    page_title="Subarachnoid Hemorrhage Analysis",
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
st.title("🧠 Subarachnoid Hemorrhage Risk Prediction")
st.markdown("""
    This tool uses transcriptome data to predict Subarachnoid Hemorrhage (SAH) risk and provides mechanistic insights 
    using SHAP (SHapley Additive exPlanations) visualizations. Adjust gene expression levels in the sidebar 
    to explore their impact on cerebrovascular regulation and SAH risk.
""")

# Load and prepare background data
@st.cache_data
def load_background_data():
    df = pd.read_excel('data/SAH_data.xlsx')  # Updated data file
    return df[['CCL20', 'IL1R1', 'GLUL', 'MAFB', 'ARL4C', 
              'BLOC1S2', 'CFB', 'HSPA5', 'PTX3', 'APOD',
              'VEGFA', 'ANGPTL4', 'CP', 'ITGA10', 'LDB3',
              'SLC25A33', 'ZIC1', 'AMPD2', 'FTH1', 'KCMF1']]

# Load the pre-trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('data/SAH_MODEL.h5')  # Updated model file

# Initialize data and model
background_data = load_background_data()
model = load_model()

# Default values for genes (example values, need to be adjusted based on actual data)
default_values = {
    'CCL20': 145.6, 'IL1R1': 1817.0, 'GLUL': 265.4, 'MAFB': 1012.0, 'ARL4C': 376.9,
    'BLOC1S2': 553.9, 'CFB': 224.1, 'HSPA5': 1103.0, 'PTX3': 1184.0, 'APOD': 83.09,
    'VEGFA': 538.5, 'ANGPTL4': 802.7, 'CP': 388.0, 'ITGA10': 424.1, 'LDB3': 94.41,
    'SLC25A33': 204.4, 'ZIC1': 226.4, 'AMPD2': 243.4, 'FTH1': 2111.0, 'KCMF1': 945.5
}

# Sidebar configuration
st.sidebar.header("🧬 Gene Expression Inputs")
st.sidebar.markdown("Adjust expression levels of SAH-related genes:")

# Reset button
if st.sidebar.button("Reset to Defaults", key="reset"):
    st.session_state.update(default_values)

# Dynamic three-column layout for more genes
gene_features = list(default_values.keys())
gene_values = {}
cols = st.sidebar.columns(3)

for i, gene in enumerate(gene_features):
    with cols[i % 3]:
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
    st.metric("SAH Risk Score", f"{prediction:.4f}", 
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
        **Key SAH-related Pathways:**
        - CCL20: Chemokine regulation in cerebrovascular inflammation
        - VEGFA: Angiogenesis and vascular permeability
        - HSPA5: Endoplasmic reticulum stress response
        - FTH1: Iron metabolism and oxidative stress
        """)
        importance_df = pd.DataFrame({'Gene': input_df.columns, 'SHAP Value': shap_values})
        importance_df = importance_df.sort_values('SHAP Value', ascending=False)
        st.dataframe(importance_df.style.background_gradient(cmap='coolwarm', subset=['SHAP Value']))

# Update documentation
with st.expander("📚 About This SAH Analysis", expanded=True):
    st.markdown("""
    ### Model Overview
    This deep learning model analyzes 20 key genes involved in:
    - Cerebrovascular regulation
    - Blood-brain barrier integrity
    - Inflammatory response
    - Neurovascular coupling
    
    ### SHAP Interpretation Guide
    1. **Force Plot**: Shows the push-pull effect of each gene on risk score
    2. **Waterfall Plot**: Step-by-step feature contribution visualization
    3. **Decision Plot**: Cumulative effect visualization
    4. **Mechanistic Insights**: Analysis combining SHAP values with known biological mechanisms
    """)

# Footer
st.markdown("---")
st.markdown(f"Developed for Subarachnoid Hemorrhage Research | Updated: {pd.Timestamp.now().strftime('%Y-%m-%d')}")