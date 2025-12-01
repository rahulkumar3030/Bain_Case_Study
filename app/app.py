import streamlit as st
import requests
import uuid
import pandas as pd
import logging
import yaml
from pathlib import Path
from dotenv import load_dotenv
from visualizations import plot_attrition_by_column, plot_percentage_stacked

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

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv(attrition_data_path)
    # Add grouped columns
    df["JobSatisfactionGroup"] = df["JobSatisfaction"].map({
        1: "Low (1-2)",
        2: "Low (1-2)",
        3: "OK (3)",
        4: "High (4-5)",
        5: "High (4-5)"
    })
    df["WorkLifeBalanceGroup"] = df["WorkLifeBalance"].map({
        1: "Bad (1-2)",
        2: "Bad (1-2)",
        3: "Average (3)",
        4: "Good (4-5)",
        5: "Good (4-5)"
    })
    return df

df = load_data()

# Create tabs
tab2, tab1 = st.tabs(["Ask HR Assistant", "HR Dashboard"])

# HR Dashboard Tab
with tab1:
    st.title("📊 HR Dashboard")
    st.write("Visualize insights from your data analysis and highlight major attrition trends.")

    # Select visualization type
    vis_type = st.selectbox("Select Visualization Type", ["Attrition by Column", "Percentage Stacked"])

    # Select column for visualization
    columns = ["Department", "JobRole", "TenureGroup", "PerformanceGroup", "JobSatisfactionGroup", "WorkLifeBalanceGroup"]
    selected_column = st.selectbox("Select Column", columns)

    # Generate and display the visualization
    if vis_type == "Attrition by Column":
        st.pyplot(plot_attrition_by_column(df, selected_column))
    elif vis_type == "Percentage Stacked":
        st.pyplot(plot_percentage_stacked(df, selected_column))

# Ask HR Assistant Tab
with tab2:
    st.title("💬 Ask HR Assistant")
    st.write("Interface for employees to input a question and receive AI-generated answers.")

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

    # Fixed layout for Ask HR Assistant section
    st.subheader("Chat with HR Assistant")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input at the bottom
    if prompt := st.chat_input("Ask a question about policies..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(prompt)

        # Show spinner while processing the response
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

        # Add assistant response to chat history
        st.session_state.messages.append({
            "role": "assistant",
            "content": bot_message
        })

        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(bot_message)

        # Rerun to refresh the chat display
        st.rerun()
