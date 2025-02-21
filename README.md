# Viettel Customer Service Chatbot

## Introduction
The Customer Service Chatbot is an AI-powered application designed to automate customer support processes using advanced technologies such as large language models (LLM) and retrieval-augmented generation (RAG). The system is capable of collecting data, processing natural language, and providing quick and accurate responses to customers.

## Key Features
- **Data Collection**: Uses `crawl4ai` to gather relevant data for the chatbot.
- **Large Language Model**: Utilizes `llama3.2` from `ollama` for natural and accurate responses.
- **Web Interface Deployment**: Built with `streamlit` for an easy-to-use interface.
- **RAG Technique**: Combines retrieval and text generation to enhance response accuracy.
- **Multi-language Support**: The chatbot can interact in multiple languages.

## Workflow
![Workflow Diagram](images/RAG_image.png)

## Inference
![Inference Diagram](images/inference.png)

## Installation and Running the Application

### 1. Install `ollama`
Please refer to the [Ollama official website](https://ollama.com) to install `ollama` based on your operating system.

### 2. Download `llama3.2` Model
Once `ollama` is installed, run the following command to download the model:
```bash
ollama pull llama3.2
```

### 3. Install Required Dependencies
Install the necessary dependencies from `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 4. Run the Application
Use `streamlit` to launch the application:
```bash
streamlit run app.py
```

## Contribution & Development
All contributions and suggestions are welcome! If you encounter issues or want to add new features, please create an issue or submit a pull request.