# Acme Corp HR Analytics Solution

An AI-powered HR management system designed to reduce employee attrition and improve HR operational efficiency through intelligent automation and data-driven insights.

## Overview

This solution addresses Acme Corp's challenges of high voluntary attrition rates and overwhelmed HR departments by providing:

1. **HR Assistant**: An AI-powered chatbot that handles employee queries regarding policies, leave rules, reimbursement processes, and more
2. **HR Visualizer Dashboard**: Interactive analytics dashboard providing insights into employee attrition patterns based on various attributes

## Business Context

**Client**: Acme Corp (Global mid-sized enterprise)
- **Employee Base**: 12,000+ employees across multiple regions
- **Challenge**: Above-average voluntary attrition, especially among younger and mid-level employees
- **Pain Points**: 
  - Poor work-life balance perception
  - Unclear growth paths
  - Management dissatisfaction
  - HR bottlenecks due to repetitive queries

## Project Structure

```
BAIN_CASE_STUDY/
├── app/                          # Streamlit application layer
│   ├── app_assistant.py          # HR Assistant chatbot interface
│   └── app.py                    # Main application entry point
│
├── data/                         # Local storage and databases
│   ├── chroma_db/                # Vector database for document embeddings
│   ├── conversation.json         # Chat conversation history
│   └── employee_attrition.csv    # Employee attrition dataset
│
├── notebooks/                    # Analytics and visualization
│   ├── analysis_utils.py         # Utility functions for data analysis
│   └── charts.ipynb              # Jupyter notebook for HR visualizer
│
├── processed_docs/               # Archive for embedded documents
│   └── [Documents moved here after processing]
│
├── rag_chatbot/                  # RAG architecture backend (FastAPI)
│   ├── models/                   # Data models and schemas
│   ├── src/                      # Core source code
│   └── utils/                    # Helper utilities
│
├── supporting_docs/              # Document repository for RAG
│   └── [HR policies, manuals, and reference documents]
│
├── .env                          # Environment variables (API keys)
├── config.yaml                   # Configuration parameters
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Features

### HR Assistant
- Natural language query processing using RAG (Retrieval-Augmented Generation) architecture
- Instant responses to policy questions, leave rules, and reimbursement procedures
- Context-aware conversations with memory retention
- Document-grounded responses ensuring accuracy

### HR Visualizer Dashboard
- Interactive attrition analytics across multiple dimensions
- Employee demographic insights
- Trend analysis and pattern identification
- Visual representations for data-driven decision making

## Technology Stack

- **Frontend**: Streamlit
- **Backend**: FastAPI
- **AI/ML**: Hybrid RAG architecture with vector embeddings
- **Vector Database**: ChromaDB
- **Data Processing**: Python, Pandas
- **Visualization**: Jupyter Notebooks, plotting libraries

## Prerequisites

- Python 3.13 with pip
- Jupyter extensions
- LLM for chat completions
- Embedding model for vector creation

## Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd BAIN_CASE_STUDY
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the root directory with the following variables:
```env
#AZURE OPEN AI SECRETS
AZURE_OPENAI_KEY='your_azure_open_ai_key'
AZURE_OPENAI_ENDPOINT='your_azure_open_ai_endpoint'
AZURE_OPENAI_DEPLOYMENT='your_deployment_model' or gpt-4o-mini

#AZURE EMBEDDING SECRETS
AZURE_EMBEDDING_KEY='your_embedding_key'
AZURE_EMBEDDING_ENDPOINT='your_embedding_endpoint'
AZURE_EMBEDDING_DEPLOYMENT=text-embedding-3-small

#YAML CONFIGURATION
CONFIG_YAML_PATH=config.yaml
```

### 5. Update Configuration
Review and update `config.yaml` with appropriate paths and parameters for your environment.

## Setup Guide

### Document Processing (First-Time Setup)
1. Place all HR policy documents, manuals, and reference materials in the `supporting_docs/` folder
2. Run the document processor to embed and store documents in the vector database:
   ```bash
   python rag_chatbot/src/document_processor.py
   ```
3. Processed documents will automatically move to `processed_docs/` folder

### Vector Database Initialization
The ChromaDB vector database will be automatically initialized in the `data/chroma_db/` directory when you first run the document processor.

## How to Run

### Run Complete Application (Recommended)
```bash
# Start the main Streamlit application
streamlit run app/app.py
```
This will launch the integrated application with both HR Assistant and Visualizer dashboards.

Access the application at `http://localhost:8501`

#### Starting the FastAPI Backend (RAG Server)
In another terminal 

```bash
# From the project root directory
python rag_chatbot/src/main.py
```
API will be available at `http://localhost:8000`


#### Running the HR Visualizer Dashboard Notebook
Open the Jupyter notebook:
```bash
jupyter notebook notebooks/charts.ipynb
```

## Usage

### HR Assistant
1. Launch the application
2. Type your query in natural language (e.g., "What is the leave policy?", "How do I claim reimbursements?")
3. Receive instant, document-grounded responses
4. Conversation history is maintained for context
5. Ask follow-up questions naturally

### HR Visualizer
1. Access the dashboard
2. Select analysis parameters:
   - Department
   - Age group
   - Tenure
   - Job role
   - Satisfaction levels
3. View interactive charts and insights
4. Identify attrition patterns and risk factors

## File Explanations

### Core Application Files
- **app/app_assistant.py**: Contains the Streamlit interface only for the HR chatbot assistant
- **app/app.py**: Main application orchestrator and entry point. Contains the Streamlit interface both Assistant and Visualizer

### Data Files
- **data/employee_attrition.csv**: Historical employee data with attrition indicators and demographics
- **data/conversation.json**: Stores chat conversation history for continuity and context
- **data/chroma_db/**: Vector database storing embedded documents for semantic search

### RAG Architecture
- **rag_chatbot/**: Complete FastAPI implementation of the Retrieval-Augmented Generation system
  - **models/**: Pydantic models for request/response validation
  - **src/**: Core application logic, document processing, and query handling
  - **utils/**: Helper functions for embeddings, retrieval, and LLM integration

### Analytics
- **notebooks/analysis_utils.py**: Reusable functions for statistical analysis and data transformation used with streamlit
- **notebooks/charts.ipynb**: Interactive notebook for exploring attrition patterns and generating visualizations

### Configuration Files
- **.env**: Stores sensitive API keys and credentials
- **config.yaml**: Non-sensitive configuration parameters (database paths, model settings, chunk sizes, etc.)
- **requirements.txt**: All Python package dependencies with version specifications. Some are without versions for auto download of other compatible dependencies

### Document Folders
- **supporting_docs/**: Place new HR documents here for embedding
- **processed_docs/**: Archive of documents that have been successfully embedded


## Future Enhancements

- Improved models for better, accurate and faster responses
- Implement Rerankers to ensure more relevant chunks are given more priority
- Integrate LLM with HR Visualizer to get insights automatically instead of harcoded
- Custom upload of attrition files

- You may reach out to me at rahul3a67@gmail.com for any issues/discussions. 
