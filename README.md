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
- **AI/ML**: RAG architecture with vector embeddings
- **Vector Database**: ChromaDB
- **Data Processing**: Python, Pandas
- **Visualization**: Jupyter Notebooks, plotting libraries

## Prerequisites

- Python 3.8 or higher
- pip package manager
- API keys for LLM services (OpenAI, Anthropic, etc.)
- Minimum 4GB RAM recommended

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
# LLM API Keys
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here  # If using Claude

# Other Configuration (if needed)
CHROMA_DB_PATH=./data/chroma_db
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

### Option 1: Run Complete Application (Recommended)
```bash
# Start the main Streamlit application
streamlit run app/app.py
```
This will launch the integrated application with both HR Assistant and Visualizer dashboards.

Access the application at `http://localhost:8501`

### Option 2: Run Components Separately

#### Starting the FastAPI Backend (RAG Server)
```bash
# From the project root directory
cd rag_chatbot
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```
API will be available at `http://localhost:8000`

#### Starting the HR Assistant
```bash
streamlit run app/app_assistant.py
```
Access at `http://localhost:8501`

#### Running the HR Visualizer Dashboard
Open the Jupyter notebook:
```bash
jupyter notebook notebooks/charts.ipynb
```
Or run via Streamlit (if integrated in app.py)

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
5. Export reports as needed

## File Explanations

### Core Application Files
- **app/app_assistant.py**: Contains the Streamlit interface for the HR chatbot assistant
- **app/app.py**: Main application orchestrator and entry point

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
- **notebooks/analysis_utils.py**: Reusable functions for statistical analysis and data transformation
- **notebooks/charts.ipynb**: Interactive notebook for exploring attrition patterns and generating visualizations

### Configuration Files
- **.env**: Stores sensitive API keys and credentials (never commit to version control)
- **config.yaml**: Non-sensitive configuration parameters (database paths, model settings, chunk sizes, etc.)
- **requirements.txt**: All Python package dependencies with version specifications

### Document Folders
- **supporting_docs/**: Place new HR documents here for embedding
- **processed_docs/**: Archive of documents that have been successfully embedded

## Troubleshooting

### Common Issues

**Issue**: ChromaDB initialization error
```bash
# Solution: Clear and reinitialize the database
rm -rf data/chroma_db
python rag_chatbot/src/document_processor.py
```

**Issue**: API key errors
- Verify `.env` file exists in root directory
- Check API keys are valid and not expired
- Ensure no extra spaces or quotes around keys

**Issue**: Port already in use
```bash
# Change the port number
streamlit run app/app.py --server.port 8502
```

**Issue**: Missing dependencies
```bash
# Reinstall all requirements
pip install -r requirements.txt --upgrade
```

## Contributing

When adding new features or documents:
1. Add new HR documents to `supporting_docs/`
2. Run document processor to update embeddings
3. Test queries against new documents
4. Update this README if configuration changes

## Security Notes

- Never commit the `.env` file to version control
- Keep API keys secure and rotate them regularly
- Limit access to employee data according to privacy policies
- Ensure compliance with GDPR/local data protection regulations

## Future Enhancements

- Multi-language support for global employee base
- Predictive attrition modeling using ML
- Integration with HRIS systems
- Mobile application for on-the-go access
- Advanced analytics with drill-down capabilities

## License

[Specify your license here]

## Contact

For questions or support, contact: [Your contact information]

---

*This project was developed as a case study solution for enterprise HR management challenges.*
