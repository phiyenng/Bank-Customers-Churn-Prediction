"""
Bank Customer Churn Prediction - Streamlit Web App
==================================================

"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import altair as alt
import joblib
import yaml
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import project modules
import sys
sys.path.append('src')
from modules.processing import DataLoader

# Page configuration
st.set_page_config(
    page_title="Bank Customer Churn Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS
st.markdown(
    """
    <style>
    :root{
      --bg:#f6f8fb;
      --card:#ffffff;
      --muted:#6b7280;
      --accent:#4B98FF;
    }
    html, body, [class*="css"] {
      background: linear-gradient(180deg,#f6f8fb 0%,#f3f6fb 100%);
    }
    .block-container{ padding:1.25rem 1.5rem 2rem 1.5rem; }

    /* Header */
    .main-header {
        font-size: 2.5rem;
        font-weight: 300;
        color: #0f172a;
        text-align: center;
        margin-bottom: 2rem;
        border-bottom: 2px solid var(--accent);
        padding-bottom: 1rem;
        transition: transform .25s ease, box-shadow .25s ease;
    }
    .main-header:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 24px rgba(15,23,42,0.08);
    }

    /* Metric cards */
    .metric-card {
        background: var(--card);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(2,6,23,0.06);
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 6px 18px rgba(15,23,42,0.04);
        transition: transform .2s ease, box-shadow .2s ease;
    }
    .metric-card:hover{ transform: translateY(-4px); box-shadow:0 16px 32px rgba(15,23,42,0.08); }
    .metric-value { font-size: 1.8rem; font-weight: 600; color: #0f172a; margin-bottom: 0.5rem; }
    .metric-label { font-size: 0.9rem; color: var(--muted); font-weight: 500; }

    /* Sidebar */
    .stSidebar{ background: linear-gradient(180deg,#fbfdff,#f8fbff); padding-top:1rem; }
    .sidebar-title{ text-align:center; font-weight:700; font-size:1.1rem; padding:0.6rem 0; color:#0f172a; }
    .sidebar .stButton>button {
        width: 80%;
        margin: 0.3rem 0;
        text-align: center;
        border-radius: 50px;
        border: 1px solid rgba(2,6,23,0.06);
        background: white;
        font-weight: 500;
        transition: all 0.25s ease;
        box-shadow:0 4px 12px rgba(15,23,42,0.04);
    }
    .sidebar .stButton>button:hover {
        background: linear-gradient(90deg, rgba(52,110,255,0.08), rgba(52,110,255,0.02));
        color:#06356b;
        box-shadow:0 10px 24px rgba(52,110,255,0.1);
    }

    /* Buttons */
    .stButton>button:not(.sidebar .stButton>button) {
        background: linear-gradient(180deg,#fbfdff,#f8fbff);
        color: black;
        border: none;
        border-radius: 30px;
        font-weight: 600;
        transition: all 0.2s ease
        box-shadow:0 4px 12px rgba(15,23,42,0.04);
    }
    .stButton>button:hover:not(.sidebar .stButton>button) {
        background-color: #f2f2f2 !important;  
        color: #0f172a !important;           
        border: 1px solid #e0e0e0 !important; 
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
    }

    .stButton>button:focus:not(.sidebar .stButton>button) {
        background-color: #e0e0e0 !important;  /* xám đậm hơn chút */
        color: #0f172a !important;
        border: 1px solid #c9c9c9 !important;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.12);
    }

    /* Chart container */
    .chart-container { background: var(--card); border-radius: 12px; border:1px solid rgba(2,6,23,0.06); padding:1.5rem; margin:1rem 0; box-shadow:0 8px 20px rgba(15,23,42,0.04); transition:box-shadow .2s ease; }
    .chart-container:hover{ box-shadow:0 14px 32px rgba(15,23,42,0.08); }

    /* Info boxes */
    .info-box, .success-box, .warning-box {
        background: #f8f9fa;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 8px;
        color: #0f172a;
        transition: all .2s ease;
    }
    .info-box{ border-left:3px solid var(--accent); }
    .success-box{ border-left:3px solid #27ae60; }
    .warning-box{ border-left:3px solid #f39c12; }
    .info-box:hover, .success-box:hover, .warning-box:hover { transform:translateX(3px); box-shadow:0 8px 18px rgba(15,23,42,0.06); }

    /* General text */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 { color:#0f172a; font-weight:500; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load configuration
@st.cache_data
def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

# Load data
@st.cache_data
def load_data():
    """Load and cache the dataset"""
    try:
        data_loader = DataLoader(load_config()['paths'])
        original_df, test_df, combined_df = data_loader.get_data()
        return original_df, test_df, combined_df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None

# Load trained model
# Define available imbalance methods
IMBALANCE_METHODS = {
    'SMOTE': 'smote_saved_models',
    'ADASYN': 'adasyn_saved_models', 
    'SMOTE + Tomek': 'smote_tomek_saved_models',
    'SMOTE + ENN': 'smote_enn_saved_models',
    'Class Weight': 'class_weight_saved_models'
}

@st.cache_resource
def load_model(method_dir="smote_saved_models"):
    """Load the trained model for specified method"""
    try:
        model_path = f"{method_dir}/best_model_tuned_optuna.joblib"
        if Path(model_path).exists():
            return joblib.load(model_path)
        else:
            st.error(f"Model file not found: {model_path}")
            return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

@st.cache_data
def load_all_metrics():
    """Load metrics for all imbalance methods"""
    all_metrics = {}
    
    for method_name, method_dir in IMBALANCE_METHODS.items():
        try:
            # Load tuned metrics
            tuned_path = f"{method_dir}/test_metrics_optuna.csv"
            baseline_path = f"{method_dir}/test_metrics_baseline.csv"
            
            if Path(tuned_path).exists():
                tuned_df = pd.read_csv(tuned_path)
                all_metrics[method_name] = {
                    'tuned': tuned_df.iloc[0].to_dict() if len(tuned_df) > 0 else None
                }
            
            if Path(baseline_path).exists():
                baseline_df = pd.read_csv(baseline_path)
                if method_name not in all_metrics:
                    all_metrics[method_name] = {}
                all_metrics[method_name]['baseline'] = baseline_df.iloc[0].to_dict() if len(baseline_df) > 0 else None
                
        except Exception as e:
            st.warning(f"Could not load metrics for {method_name}: {str(e)}")
            
    return all_metrics

# Load test metrics
@st.cache_data
def load_metrics():
    """Load model performance metrics"""
    try:
        baseline_metrics = pd.read_csv("smote_saved_models/test_metrics_baseline.csv")
        tuned_metrics = pd.read_csv("smote_saved_models/test_metrics_optuna.csv")
        cv_metrics = pd.read_csv("smote_saved_models/cv/metrics_summary.csv")
        return baseline_metrics, tuned_metrics, cv_metrics
    except Exception as e:
        st.error(f"Error loading metrics: {str(e)}")
        return None, None, None

def main():
    # Sidebar navigation
    st.sidebar.markdown('<div class="sidebar-title">Bank Churn Prediction</div>', unsafe_allow_html=True)
    st.sidebar.markdown("---")
    
    # Initialize session state for page selection
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Project Information"
    
    if st.sidebar.button("Project Information", width='stretch', 
                        type="primary" if st.session_state.current_page == "Project Information" else "secondary"):
        st.session_state.current_page = "Project Information"
    
    if st.sidebar.button("Dashboard", width='stretch',
                        type="primary" if st.session_state.current_page == "Dashboard" else "secondary"):
        st.session_state.current_page = "Dashboard"
    
    if st.sidebar.button("Prediction", width='stretch',
                        type="primary" if st.session_state.current_page == "Prediction" else "secondary"):
        st.session_state.current_page = "Prediction"
    
    # Project Information at bottom of sidebar
    st.sidebar.markdown("<br><br>", unsafe_allow_html=True)
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="font-size: 0.7rem; color: #6b7280; text-align: left; padding: 0.5rem; margin-bottom: 0.5rem;">
        <span style="display: inline-block; margin-left: 8px;">
            <strong>Comparative Analysis of Gradient Boosting Models for Predicting Bank Customer Churn</strong><br><br>
            <strong>Author:</strong> Nguyen Van Phi Yen<br>
            <strong>Email:</strong> yennguyen.31221021785@st.ueh.edu.vn<br>
            University of Economics Ho Chi Minh City<br>
            <strong>Advisor:</strong> Ph.D. Nguyen Quoc Hung
        </span>
    </div>
    """, unsafe_allow_html=True)

    
    page = st.session_state.current_page
    
    # Load data and metrics
    original_df, test_df, combined_df = load_data()
    baseline_metrics, tuned_metrics, cv_metrics = load_metrics()
    all_metrics = load_all_metrics()
    
    if page == "Project Information":
        show_project_info(combined_df, baseline_metrics, tuned_metrics, cv_metrics, all_metrics)
    elif page == "Dashboard":
        show_dashboard(combined_df)
    elif page == "Prediction":
        show_prediction_page(combined_df)

def show_project_info(df, baseline_metrics, tuned_metrics, cv_metrics, all_metrics):
    """Display project information page"""
    st.markdown('<h1 class="main-header">Bank Customer Churn Prediction</h1>', unsafe_allow_html=True)
    
    # Project overview
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Project Overview
        
        This project focuses on predicting bank customer churn using machine learning techniques. 
        The goal is to identify customers who are likely to leave the bank so that proactive 
        retention strategies can be implemented.
        
        **Key Features:**
        - Advanced feature engineering and selection
        - Multiple ML algorithms (XGBoost, LightGBM, CatBoost)
        - Hyperparameter optimization with Optuna
        - Comprehensive evaluation metrics
        - Interactive web interface for predictions
        """)
    
    with col2:
        if df is not None:
            st.markdown("""
            ### Dataset Summary
            """)
            st.metric("Total Records", f"{len(df):,}")
            st.metric("Features", f"{len(df.columns)-1}")
            churn_rate = df['Exited'].mean() * 100
            st.metric("Churn Rate", f"{churn_rate:.1f}%")
    
    st.markdown("---")
    
    # Proposed Framework
    st.markdown("### Proposed Framework")
    
    st.markdown("""
    This research proposes a comprehensive framework for bank customer churn prediction that integrates 
    multiple imbalance handling techniques with gradient boosting algorithms. The framework consists of 
    several key components working together to achieve optimal prediction performance.
    """)
    
    # Display framework image
    try:
        st.image("framework.png", 
                caption="Proposed Framework for Bank Customer Churn Prediction", 
                use_container_width =True)
    except Exception as e:
        st.warning("Framework image not found. Please ensure 'framework.png' is in the project directory.")
    
    st.markdown("""
    **Framework Components:**
    
    1. **Data Preprocessing**: Outlier removal, duplicate handling, and encoding
    2. **Feature Engineering**: Feature creation and selection for optimal model input
    3. **Class Imbalance Handling**: Multiple techniques (SMOTE, ADASYN, Hybrid methods, Class weighting)
    4. **Model Training**: Gradient boosting algorithms (XGBoost, LightGBM, CatBoost)
    5. **Cross-Validation**: Stratified K-fold for robust evaluation
    6. **Hyperparameter Tuning**: Optuna-based optimization
    7. **Model Evaluation**: Comprehensive metrics and interpretability analysis
    """)
    
    st.markdown("---")
    
    # Model Performance
    st.markdown("### Model Performance")
    
    if tuned_metrics is not None and baseline_metrics is not None:
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.markdown('''
            <div class="metric-card">
                <div class="metric-label">Best Model</div>
                <div class="metric-value">LightGBM</div>
            </div>
            ''', unsafe_allow_html=True)

        with col2:
            st.markdown('''
            <div class="metric-card">
                <div class="metric-label">Imbalance Method</div>
                <div class="metric-value">SMOTE</div>
            </div>
            ''', unsafe_allow_html=True)
        
        with col3:
            roc_auc = tuned_metrics.iloc[0]['ROC_AUC']
            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-label">ROC-AUC</div>
                <div class="metric-value">{roc_auc:.3f}</div>
            </div>
            ''', unsafe_allow_html=True)
        
        with col4:
            accuracy = tuned_metrics.iloc[0]['Accuracy']
            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-label">Accuracy</div>
                <div class="metric-value">{accuracy:.3f}</div>
            </div>
            ''', unsafe_allow_html=True)
        
        with col5:
            f1 = tuned_metrics.iloc[0]['F1']
            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-label">F1-Score</div>
                <div class="metric-value">{f1:.3f}</div>
            </div>
            ''', unsafe_allow_html=True)
    
    # Technical Details
    st.markdown("### Technical Implementation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Data Processing:**
        - Outlier detection and removal (IQR method)
        - Feature engineering (age categories, credit score ranges)
        - Feature transformation (PowerTransformer)
        - Feature selection (correlation-based)
        - Imbalanced data handling (ADASYN)
        """)
    
    with col2:
        st.markdown("""
        **Model Training:**
        - Cross-validation (5-fold stratified)
        - Hyperparameter optimization (Optuna)
        - Multiple algorithms comparison
        - Comprehensive evaluation metrics
        - SHAP interpretability analysis
        """)
    
    # Dataset Information
    if df is not None:
        st.markdown("### Dataset Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Dataset Statistics:**")
            st.dataframe(df.describe(), width='stretch')
        
        with col2:
            st.markdown("**Data Types:**")
            st.dataframe(df.dtypes.to_frame('Data Type'), width='stretch')
    
    # Methods Comparison Table
    st.markdown("---")
    st.markdown("### Imbalance Methods Performance Comparison")
    
    if all_metrics:
        # Create comparison dataframe
        comparison_data = []
        
        for method_name, metrics in all_metrics.items():
            if 'tuned' in metrics and metrics['tuned']:
                tuned_data = metrics['tuned']
                # Extract model name from the Model column
                model_name = tuned_data.get('Model', 'Unknown')
                # Clean up model name for display
                if '_tuned_optuna' in model_name:
                    clean_model_name = model_name.replace('_tuned_optuna', '')
                elif '_baseline' in model_name:
                    clean_model_name = model_name.replace('_baseline', '')
                else:
                    clean_model_name = model_name
                
                comparison_data.append({
                    'Method': method_name,
                    'Model': clean_model_name,
                    'Accuracy': f"{tuned_data.get('Accuracy', 0):.4f}",
                    'F1 Score': f"{tuned_data.get('F1', 0):.4f}",
                    'Precision': f"{tuned_data.get('Precision', 0):.4f}",
                    'Recall': f"{tuned_data.get('Recall', 0):.4f}",
                    'ROC AUC': f"{tuned_data.get('ROC_AUC', 0):.4f}",
                    'PR AUC': f"{tuned_data.get('PR_AUC', 0):.4f}"
                })
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            
            # Find best method for each metric
            numeric_df = comparison_df.copy()
            for col in ['Accuracy', 'F1 Score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']:
                numeric_df[col] = pd.to_numeric(numeric_df[col])
            
            # Style the dataframe to highlight best values
            def highlight_best(s):
                if s.name in ['Accuracy', 'F1 Score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']:
                    max_val = s.max()
                    return ['background-color: #e8f5e8; font-weight: bold' if v == max_val else '' for v in s]
                return ['' for _ in s]
            
            styled_df = comparison_df.style.apply(highlight_best, axis=0)
            
            st.dataframe(styled_df, width='stretch', hide_index=True)
            
            # Best method summary
            best_roc_idx = numeric_df['ROC AUC'].idxmax()
            best_roc_method = numeric_df.loc[best_roc_idx, 'Method']
            best_roc_model = comparison_df.loc[best_roc_idx, 'Model']
            best_roc_score = numeric_df['ROC AUC'].max()
            
            st.markdown(f"""
            **🏆 Best Performing Method:** {best_roc_method} with {best_roc_model} (ROC AUC: {best_roc_score:.4f})
            
            **Key Insights:**
            - All methods show competitive performance
            - ROC AUC is the primary metric for imbalanced classification
            - Different methods may excel in different scenarios
            - Model: {best_roc_model} consistently performs well across methods
            """)
        else:
            st.warning("No tuned metrics found for comparison")
    else:
        st.warning("Could not load metrics for comparison")

def show_dashboard(df):
    """Display interactive dashboard"""
    st.markdown('<h1 class="main-header">Data Dashboard</h1>', unsafe_allow_html=True)
    
    if df is None:
        st.error("Unable to load data for dashboard")
        return
    
    # Interactive filters
    st.markdown("### Filters")
    
    # Create filter columns
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    
    with filter_col1:
        GEOGRAPHIES = df["Geography"].unique()
        selected_geographies = st.pills(
            "Countries", GEOGRAPHIES, default=GEOGRAPHIES, selection_mode="multi"
        )
    
    with filter_col2:
        GENDERS = df["Gender"].unique()
        selected_genders = st.pills(
            "Gender", GENDERS, default=GENDERS, selection_mode="multi"
        )
    
    with filter_col3:
        # Age range slider
        min_age = int(df["Age"].min())
        max_age = int(df["Age"].max())
        age_range = st.slider(
            "Age Range",
            min_value=min_age,
            max_value=max_age,
            value=(min_age, max_age),
            step=1,
            help="Select age range to analyze"
        )
    
    # Validation
    if not selected_geographies:
        st.warning("You must select at least 1 country.", icon=":material/warning:")
        return
    
    if not selected_genders:
        st.warning("You must select at least 1 gender.", icon=":material/warning:")
        return
    
    # Filter data based on all selections
    filtered_df = df[
        (df["Geography"].isin(selected_geographies)) &
        (df["Gender"].isin(selected_genders)) &
        (df["Age"] >= age_range[0]) &
        (df["Age"] <= age_range[1])
    ]
    
    # Show filtered data info
    st.info(f"Showing {len(filtered_df):,} customers out of {len(df):,} total customers ({len(filtered_df)/len(df)*100:.1f}%)")
    
    if len(filtered_df) == 0:
        st.warning("No data matches the selected filters. Please adjust your selection.")
        return
    
    # Key Metrics Row
    st.markdown("### Key Metrics")
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        total_customers = len(filtered_df)
        st.metric("Total Customers", f"{total_customers:,}")
    
    with col2:
        churn_rate = filtered_df['Exited'].mean() * 100
        st.metric("Churn Rate", f"{churn_rate:.1f}%")
    
    with col3:
        avg_age = filtered_df['Age'].mean()
        st.metric("Avg Age", f"{avg_age:.1f}")
    
    with col4:
        avg_balance = filtered_df['Balance'].mean()
        st.metric("Avg Balance", f"${avg_balance:,.0f}")
    
    with col5:
        avg_credit_score = filtered_df['CreditScore'].mean()
        st.metric("Avg Credit Score", f"{avg_credit_score:.0f}")
    
    with col6:
        active_members = filtered_df['IsActiveMember'].mean() * 100
        st.metric("Active Members", f"{active_members:.1f}%")
    
    st.markdown("---")
    
    # Main dashboard layout
    cols = st.columns([3, 1])

    with cols[0].container(border=True, height="stretch"):
        st.markdown("### Churn by Gender")
        
        grouped = (
            filtered_df.groupby(['Gender', 'Exited'])
            .size()
            .reset_index(name='Count')
        )

        exited_map = {0: "Stayed", 1: "Churned"}
        grouped['Exited_Label'] = grouped['Exited'].map(exited_map)

        chart = (
            alt.Chart(grouped)
            .mark_bar()
            .encode(
                x=alt.X('Gender:N', title='Gender', axis=alt.Axis(labelAngle=0)),  # chữ ngang
                y=alt.Y('Count:Q', title='Number of Customers'),
                color=alt.Color(
                    'Exited_Label:N',
                    title='Churn Status',
                    scale=alt.Scale(range=['#83C9FFff', '#FFABABff'])
                ),
                xOffset='Exited_Label:N'  # grouped bar
            )
            .properties(width=alt.Step(60))
            .configure_axis(
                labelFontSize=12,
                titleFontSize=14
            )
            .configure_legend(
                orient="bottom"
            )
        )

        st.altair_chart(chart, use_container_width=True)

    
    
    with cols[1].container(border=True, height="stretch"):
        st.markdown("### Churn Distribution")
        
        # Map Exited to labels for legend clarity
        exited_map = {0: "Stayed", 1: "Churned"}
        filtered_df_plot = filtered_df.copy()
        filtered_df_plot['Exited_Label'] = filtered_df_plot['Exited'].map(exited_map)
        
        st.altair_chart(
            alt.Chart(filtered_df_plot)
            .mark_arc()
            .encode(
                alt.Theta("count()"),
                alt.Color("Exited_Label:N")
                    .scale(domain=["Stayed", "Churned"], range=['#0068C9ff', '#83C9FFff'])
                    .title("Status"),
            )
            .configure_legend(orient="bottom"),
            use_container_width=True
        )
    
    # Second row
    cols = st.columns(2)
    
    with cols[0].container(border=True, height="stretch"):
        "### Credit Score Distribution"
        
        st.altair_chart(
            alt.Chart(filtered_df)
            .mark_bar(opacity=0.7)
            .encode(
                alt.X("CreditScore:Q").bin(maxbins=30).title("Credit Score"),
                alt.Y("count():Q").title("Number of Customers"),
                alt.Color("Exited:N").scale(range=['#29B09D', '#7DEFA1']).title("Churn Status"),
            )
            .configure_legend(orient="bottom"),
            use_container_width=True
        )
    
    with cols[1].container(border=True, height="stretch"):
        st.markdown("### Geographic Churn Distribution")
        geo_churn_counts = (
            filtered_df.groupby(['Geography', 'Exited'])
            .size()
            .reset_index(name='Count')
        )

        exited_map = {0: "Stayed", 1: "Churned"}
        geo_churn_counts['Exited_Label'] = geo_churn_counts['Exited'].map(exited_map)

        chart = (
            alt.Chart(geo_churn_counts)
            .mark_bar()
            .encode(
                x=alt.X('Geography:N', title='Geography', axis=alt.Axis(labelAngle=0)),
                y=alt.Y('Count:Q', title='Number of Customers'),
                color=alt.Color(
                    'Exited_Label:N',
                    title='Churn Status',
                    scale=alt.Scale(range=['#0068C9', '#83C9FF'])
                ),
                xOffset='Exited_Label:N' 
            )
            .properties(width=alt.Step(60))
            .configure_axis(
                labelFontSize=12,
                titleFontSize=14
            )
            .configure_legend(
                orient="bottom"
            )
        )

        st.altair_chart(chart, use_container_width=True)
        
    # Third row
    cols = st.columns(2)
    
    with cols[0].container(border=True, height="stretch"):
        "### Salary vs Balance Correlation"
        
        st.altair_chart(
            alt.Chart(filtered_df.sample(min(1500, len(filtered_df))))
            .mark_circle(size=40, opacity=0.6)
            .encode(
                alt.X("EstimatedSalary:Q").title("Estimated Salary ($)"),
                alt.Y("Balance:Q").title("Account Balance ($)"),
                alt.Color("Geography:N").scale(scheme='category10').title("Country"),
                alt.Size("Exited:N").scale(range=[30, 100]).title("Churned"),
                alt.Tooltip(['EstimatedSalary:Q', 'Balance:Q', 'Geography:N', 'Exited:N'])
            )
            .configure_legend(orient="bottom"),
            use_container_width=True
        )
    
    with cols[1].container(border=True, height="stretch"):
        "### Age vs Balance Analysis"
        
        # Create age groups for better visualization
        age_bins = [18, 30, 40, 50, 60, 100]
        age_labels = ['18-30', '31-40', '41-50', '51-60', '60+']
        filtered_df_copy = filtered_df.copy()
        filtered_df_copy['AgeGroup'] = pd.cut(filtered_df_copy['Age'], bins=age_bins, labels=age_labels, right=False)
        
        st.altair_chart(
            alt.Chart(filtered_df_copy.sample(min(2000, len(filtered_df_copy))))
            .mark_circle(size=60, opacity=0.6)
            .encode(
                alt.X("Age:Q").title("Age"),
                alt.Y("Balance:Q").title("Account Balance ($)"),
                alt.Color("Exited:N").scale(range=['#3498db', '#e74c3c']).title("Churn Status"),
                alt.Tooltip(['Age:Q', 'Balance:Q', 'Geography:N', 'Exited:N'])
            )
            .configure_legend(orient="bottom")
            .resolve_scale(color='independent'),
            use_container_width=True
        )
    
    # Fourth row
    cols = st.columns(1)

    with cols[0].container(border=True, height="stretch"):
        st.markdown("### Correlation Matrix - Pearson")

        # Select only numeric columns for correlation
        numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
        corr_matrix = filtered_df[numeric_cols].corr(method='pearson')

        # Prepare data for heatmap
        corr_matrix_reset = corr_matrix.reset_index().melt(id_vars='index')
        corr_matrix_reset.columns = ['Feature1', 'Feature2', 'Correlation']

        heatmap = alt.Chart(corr_matrix_reset).mark_rect().encode(
            x=alt.X('Feature1:O', sort=numeric_cols, title=None),
            y=alt.Y('Feature2:O', sort=numeric_cols, title=None),
            color=alt.Color('Correlation:Q', scale=alt.Scale(scheme='redblue', domain=[-1, 1])),
            tooltip=['Feature1', 'Feature2', alt.Tooltip('Correlation:Q', format=".2f")]
        ).properties(
            width=300,
            height=300
        )

        text = alt.Chart(corr_matrix_reset).mark_text(baseline='middle').encode(
            x=alt.X('Feature1:O', sort=numeric_cols),
            y=alt.Y('Feature2:O', sort=numeric_cols),
            text=alt.Text('Correlation:Q', format=".2f"),
            color=alt.condition(
                "abs(datum.Correlation) > 0.5",
                alt.value('white'),
                alt.value('black')
            )
        )

        st.altair_chart(heatmap + text, use_container_width=True)

def show_prediction_page(df):
    """Display prediction interface"""
    st.markdown('<h1 class="main-header">Customer Churn Prediction</h1>', unsafe_allow_html=True)
    
    # Method selection
    st.markdown("### 🎯 Select Imbalance Handling Method")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_method = st.selectbox(
            "Choose the imbalance method for prediction:",
            options=list(IMBALANCE_METHODS.keys()),
            index=0,
            help="Different methods may give slightly different predictions based on how they handle class imbalance during training."
        )
    
    with col2:
        st.markdown(f"""
        **Selected Method:** {selected_method}
        
        **Method Info:**
        - SMOTE: Synthetic Minority Oversampling
        - ADASYN: Adaptive Synthetic Sampling  
        - SMOTE + Tomek: Hybrid approach
        - SMOTE + ENN: Edited Nearest Neighbours
        - Class Weight: Weighted training
        """)
    
    # Load selected model
    method_dir = IMBALANCE_METHODS[selected_method]
    model = load_model(method_dir)
    
    if model is None:
        st.error(f"Model not available for {selected_method}. Please check if the model file exists.")
        return
    
    st.markdown("""
    ### Enter Customer Information
    
    Please fill in the customer details below to predict their churn probability.
    """)
    
    # Create input form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Personal Information")
            credit_score = st.slider("Credit Score", 300, 850, 650, help="Customer's credit score")
            age = st.slider("Age", 18, 100, 35, help="Customer's age")
            gender = st.selectbox("Gender", ["Male", "Female"], help="Customer's gender")
            geography = st.selectbox("Geography", ["France", "Germany", "Spain"], help="Customer's country")
            
            st.markdown("#### Financial Information")
            balance = st.number_input("Balance", 0.0, 250000.0, 0.0, step=1000.0, help="Customer's account balance")
            estimated_salary = st.number_input("Estimated Salary", 0.0, 200000.0, 50000.0, step=1000.0, help="Customer's estimated salary")
        
        with col2:
            st.markdown("#### Banking Information")
            tenure = st.slider("Tenure (Years)", 0, 10, 5, help="Number of years as customer")
            num_products = st.selectbox("Number of Products", [1, 2, 3, 4], help="Number of bank products used")
            has_credit_card = st.selectbox("Has Credit Card", ["Yes", "No"], help="Whether customer has credit card")
            is_active_member = st.selectbox("Is Active Member", ["Yes", "No"], help="Whether customer is active")
        
        # Submit button
        submitted = st.form_submit_button("Predict Churn Probability", width='stretch')
    
    if submitted:
        # Prepare input data
        input_data = {
            'CreditScore': credit_score,
            'Geography': geography,
            'Gender': gender,
            'Age': age,
            'Tenure': tenure,
            'Balance': balance,
            'NumOfProducts': num_products,
            'HasCrCard': 1 if has_credit_card == "Yes" else 0,
            'IsActiveMember': 1 if is_active_member == "Yes" else 0,
            'EstimatedSalary': estimated_salary
        }
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        
        try:
            # Apply feature engineering (simplified version)
            input_df = apply_feature_engineering(input_df)
            
            # Make prediction
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0]
            
            # Display results
            st.markdown("---")
            st.markdown(f"### 🎯 Prediction Results - {selected_method}")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                churn_prob = probability[1] * 100
                st.metric("Churn Probability", f"{churn_prob:.1f}%")
            
            with col2:
                stay_prob = probability[0] * 100
                st.metric("Stay Probability", f"{stay_prob:.1f}%")
            
            with col3:
                prediction_text = "HIGH RISK" if prediction == 1 else "LOW RISK"
                st.metric("Risk Level", prediction_text)
            
            with col4:
                st.metric("Method Used", selected_method)
            
            # Risk assessment
            if churn_prob > 70:
                st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                st.markdown("**High Risk Customer** - Immediate attention required!")
                st.markdown("</div>", unsafe_allow_html=True)
            elif churn_prob > 40:
                st.markdown('<div class="info-box">', unsafe_allow_html=True)
                st.markdown("**Medium Risk Customer** - Monitor closely")
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.markdown('<div class="success-box">', unsafe_allow_html=True)
                st.markdown("**Low Risk Customer** - Likely to stay")
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Recommendations
            st.markdown("### Recommendations")
            
            if prediction == 1:
                st.markdown("""
                **For High-Risk Customers:**
                - Offer personalized retention incentives
                - Schedule immediate follow-up call
                - Review account activity and address concerns
                - Consider offering premium services or discounts
                """)
            else:
                st.markdown("""
                **For Low-Risk Customers:**
                - Continue current service level
                - Regular check-ins to maintain satisfaction
                - Consider upselling additional products
                - Monitor for any changes in behavior
                """)
        
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

def apply_feature_engineering(df):
    """Apply feature engineering to input data"""
    # This is a simplified version - in practice, you'd load the fitted transformers
    df = df.copy()
    
    # Age categories
    df['Age_Category'] = pd.cut(
        df['Age'],
        bins=[18, 30, 40, 50, 60, 100],
        labels=['18-30', '30-40', '40-50', '50-60', '60+'],
        include_lowest=True
    )
    
    # Credit score ranges
    df['Credit_Score_Range'] = pd.cut(
        df['CreditScore'],
        bins=[0, 300, 600, 700, 800, 900],
        labels=['0-300', '300-600', '600-700', '700-800', '900+'],
        include_lowest=True
    )
    
    # One-hot encoding for categorical variables
    df = pd.get_dummies(df, columns=['Geography', 'Gender', 'Age_Category', 'Credit_Score_Range'], drop_first=True)
    
    # Add missing columns that might be in the trained model
    expected_columns = [
        'Balance', 'EstimatedSalary', 'Tenure', 'Age', 'NumOfProducts', 'IsActiveMember', 
        'Gender', 'HasCrCard', 'CreditScore', 'Total_Products_Used', 'Geography_Germany', 
        'Geography_Spain', 'Age_Category_30-40', 'Age_Category_40-50', 'Age_Category_50-60',
        'Credit_Score_Range_600-700', 'Credit_Score_Range_700-800', 'Credit_Score_Range_900+',
        'Geo_Gender_France_Male', 'Geo_Gender_Germany_Female', 'Geo_Gender_Germany_Male',
        'Geo_Gender_Spain_Female', 'Geo_Gender_Spain_Male', 'Tp_Gender_1.0_Male',
        'Tp_Gender_2.0_Female', 'Tp_Gender_2.0_Male', 'Tp_Gender_3.0_Female',
        'Tp_Gender_3.0_Male', 'Tp_Gender_4.0_Female', 'Tp_Gender_4.0_Male'
    ]
    
    # Add missing columns with default values
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0
    
    # Ensure correct order of columns
    df = df.reindex(columns=expected_columns, fill_value=0)
    
    return df

if __name__ == "__main__":
    main()
