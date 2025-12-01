import sys
import streamlit as st
import requests
import uuid
import pandas as pd
import logging
import yaml
from pathlib import Path
from dotenv import load_dotenv
sys.path.append(str(Path(__file__).resolve().parent.parent))
# Import analysis functions
from notebooks.analysis_utils import (
    create_attrition_plot, 
    get_insight_text, 
    COLUMN_DISPLAY_NAMES,
    GROUPING_CONFIG
)

# FastAPI backend URL
FASTAPI_URL = "http://localhost:8000"

# Page configuration
st.set_page_config(
    page_title="HR Dashboard & Assistant",
    page_icon="💼",
    layout="wide"
)

# Load config, loggers, and set paths
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
base_dir = Path(__file__).resolve().parent.parent
config_file_path = Path(base_dir, "config.yaml")

if not config_file_path.exists():
    logger.error(f"Configuration file not found: {config_file_path}")
    raise FileNotFoundError(f"Configuration file not found: {config_file_path}")

try:
    with open(config_file_path, 'r') as f:
        config = yaml.safe_load(f)
except yaml.YAMLError as e:
    logger.error(f"Error parsing YAML configuration file: {e}")
    raise

logger.info(f"Configuration loaded from {config_file_path}: {config}")
attrition_data_path = Path(base_dir, config['data_paths']['attrition_data'])

if not attrition_data_path.exists():
    logger.error(f"Attrition data file not found: {attrition_data_path}")
    raise FileNotFoundError(f"Attrition data file not found: {attrition_data_path}")

logger.info(f"Attrition data path set to: {attrition_data_path}")

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv(attrition_data_path)
    return df

df = load_data()

# Create tabs
tab2, tab1 = st.tabs(["Ask HR Assistant", "HR Dashboard"])

# ========================================
# HR Dashboard Tab (VISUALIZER)
# ========================================
with tab1:
    st.title("📊 HR Attrition Visualizer")
    st.write("Analyze employee attrition patterns by selecting different dimensions.")
    
    # Create column selection
    available_columns = [
        'JobSatisfaction',
        'OverTime',
        'YearsAtCompany',
        'WorkLifeBalance',
        'PerformanceRating',
        'Department',
        'JobRole'
    ]
    
    # Create friendly display names for dropdown
    column_options = {COLUMN_DISPLAY_NAMES.get(col, col): col for col in available_columns}
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        selected_display_name = st.selectbox(
            "Select dimension to analyze:",
            options=list(column_options.keys()),
            index=0
        )
    
    with col2:
        # Show top N option for high-cardinality columns
        selected_column = column_options[selected_display_name]
        if selected_column in ['JobRole', 'Department']:
            top_n = st.number_input("Show top N categories:", min_value=5, max_value=20, value=10)
        else:
            top_n = None
    
    st.divider()
    
    # Create visualization
    try:
        fig, summary = create_attrition_plot(df, selected_column, top_n)
        
        # Display plot
        st.pyplot(fig)
        
        # Display insights
        st.subheader("📌 Key Insights")
        insight_text = get_insight_text(selected_column)
        st.info(insight_text)
        
        # Display summary table
        with st.expander("📊 View Detailed Statistics"):
            st.dataframe(
                summary,
                use_container_width=True,
                hide_index=True
            )
            
            # Summary metrics
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Highest Attrition", 
                         f"{summary.iloc[0]['Category']}", 
                         f"{summary.iloc[0]['Attrition_Rate']:.1f}%")
            with col_b:
                st.metric("Lowest Attrition", 
                         f"{summary.iloc[-1]['Category']}", 
                         f"{summary.iloc[-1]['Attrition_Rate']:.1f}%")
            with col_c:
                avg_attrition = (df['Attrition'] == 'Yes').sum() / len(df) * 100
                st.metric("Overall Attrition", f"{avg_attrition:.2f}%")
        
    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")
        logger.error(f"Visualization error: {e}")

# ========================================
# Ask HR Assistant Tab
# ========================================
with tab2:
    # Initialize session state
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Sidebar - Session controls
    with st.sidebar:
        st.header("Session Info")
        st.text_input("Session ID", value=st.session_state.session_id, disabled=True)
        
        if st.button("🔄 New Session"):
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.rerun()
        
        st.divider()
        st.caption(f"Total messages: {len(st.session_state.messages)}")

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Chat input OUTSIDE the tab container
if prompt := st.chat_input("Ask a question about policies..."):
    # Add user message to chat history FIRST
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Show spinner while processing
    with st.spinner("Thinking..."):
        try:
            # Make POST request to FastAPI backend
            response = requests.post(
                f"{FASTAPI_URL}/chats",
                json={
                    "session_id": st.session_state.session_id,
                    "user_message": prompt
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                bot_message = data["bot_message"]
            else:
                bot_message = f"Error: {response.status_code} - {response.text}"
                
        except requests.exceptions.ConnectionError:
            bot_message = "⚠️ Cannot connect to FastAPI server. Make sure it's running on http://localhost:8000"
        except requests.exceptions.Timeout:
            bot_message = "⏱️ Request timed out. Please try again."
        except Exception as e:
            bot_message = f"❌ An error occurred: {str(e)}"
    
    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(bot_message)
    
    # Add assistant response to chat history
    st.session_state.messages.append({
        "role": "assistant",
        "content": bot_message
    })
    
    # Rerun to refresh the chat display
    st.rerun()
