# IBM RAG Lab with LangChain and IBM Granite

This repository contains the completed Retrieval-Augmented Generation (RAG) lab from the IBM Granite Snack Cookbook: **RAG with LangChain**. 

It has been expanded to support multiple LLM backends:
1. **Ollama** (Runs locally and is 100% free, using `qwen2.5:1.5b` or `granite-code`)
2. **IBM watsonx.ai** (Programmatic access to IBM's Granite Models in the cloud)
3. **Replicate** (Cloud-hosted serverless Granite API)

---

## 🛠️ Step-by-Step Setup Guide

### Task 1: Sign up for an IBM watsonx trial account
If you want to use the cloud watsonx.ai Granite models:
1. Visit the IBM watsonx page: [https://www.ibm.com/products/watsonx](https://www.ibm.com/products/watsonx).
2. Click **Try watsonx for free**.
3. Select your region (e.g., **Dallas**) and create an account or log in with your existing IBMid.
4. Follow the prompt to provision your Sandbox project.

### Task 2: Obtain your credentials for programmatic access
To configure API access:
1. Navigate back to the watsonx home screen and scroll down to the **Developer access** section.
2. In the *Project or space* drop-down, select your **Sandbox project**.
3. Copy the **Project ID** and save it.
4. Copy the **watsonx.ai URL** (e.g., `https://us-south.ml.cloud.ibm.com`) and save it.
5. Click **Create API key**:
   - Give it a name like `watsonx API key`.
   - Copy the API key and download the JSON file for your records.

### Task 3: Associate the watsonx.ai Runtime service with your project
Ensure your Sandbox project has access to run models:
1. Go to the watsonx home page and click on your Sandbox project under **Projects**.
2. Click the **Manage** tab at the top.
3. Select **Services & integrations** from the left panel.
4. Click **Associate service**.
5. Select your **watsonx.ai Runtime** service and click **Associate**.

---

## 🚀 Running the Project Locally

### 1. Prerequisites
- Python 3.11+ (CPython 3.11.15 is configured in the virtual environment).
- [Ollama](https://ollama.com/) (Optional - only if running fully offline).

### 2. Set Up Virtual Environment
Create and activate the environment, then install requirements:
```bash
# Create environment
python -m venv .venv

# Activate environment
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration
Copy `.env.example` to `.env` and fill in your credentials:
```bash
cp .env.example .env
```
Open `.env` and configure:
- For **Ollama**: No keys needed. Make sure the Ollama desktop app is running and your model is downloaded (e.g., `ollama pull qwen2.5:1.5b` or `ollama pull granite-code:8b`).
- For **watsonx.ai**: Fill in `WATSONX_APIKEY`, `WATSONX_PROJECT_ID`, and `WATSONX_URL`.
- For **Replicate**: Fill in `REPLICATE_API_TOKEN`.

---

## 💻 Running the Code

### Option A: Run the Python CLI Script
You can run the RAG pipeline directly from the command line:

```bash
# 1. Run using local Ollama (Default)
python rag_with_langchain.py --provider ollama

# 2. Run using IBM watsonx.ai Cloud
python rag_with_langchain.py --provider watsonx

# 3. Run using Replicate (as in the original Colab recipe)
python rag_with_langchain.py --provider replicate

# Pass a custom question:
python rag_with_langchain.py --provider ollama --query "What is the President's plan for inflation?"
```

### Option B: Run the Jupyter Notebook
Start the Jupyter environment to step through the code visually:
```bash
jupyter notebook RAG_with_Langchain.ipynb
```
The notebook is fully configured and annotated, allowing you to run each block individually.

---

## 📁 Repository Structure

- `RAG_with_Langchain.ipynb`: Completed and modified Jupyter Notebook from the original IBM Granite Snack Cookbook.
- `rag_with_langchain.py`: Production-ready Python script implementing the RAG pipeline with multiple provider options.
- `.env.example` & `.env`: Configurations for API keys.
- `.gitignore`: Prevents temporary files, caches, databases, and API keys from leaking to GitHub.
