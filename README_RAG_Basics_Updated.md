# 🧠 RAG Basics — Multi-PDF Question Answering using LangChain, Groq, and FAISS

This repository demonstrates a **Retrieval-Augmented Generation (RAG)** pipeline for reading and querying multiple PDFs locally using **LangChain**, **HuggingFace embeddings**, and **Groq LLMs**.

You can use this code to load PDFs, chunk them into embeddings, store them in a FAISS vector database, and query them using natural language.

---

## 📚 Overview

**Goal:** Build a local RAG pipeline capable of understanding multiple PDFs and answering questions contextually.

### ⚙️ Main Components
| Component | Description |
|------------|-------------|
| **LangChain** | Framework to manage document loaders, retrievers, and chains |
| **HuggingFace Embeddings** | Converts text chunks into numerical vectors |
| **FAISS** | Vector store for fast semantic search |
| **Groq LLM (openai/gpt-oss-120b)** | Model that generates final responses |
| **RAG Pipeline** | Combines retrieval (from FAISS) and generation (from LLM) |

---

## 🛠️ Setup Instructions

### 🧩 Folder Structure

```
RAG/
│
├── rag_basics.py              # Main Python script
├── documents/                 # Folder containing your PDFs
├── local_secrets.py (optional)# File to safely store your API keys locally
└── README.md
```

### 📦 Step 1 — Create Virtual Environment

```bash
# Navigate to your project folder
cd "C:\Users\saish\OneDrive\Desktop\RAG"

# Create venv (note the space)
python -m venv .venv

# Activate (PowerShell)
.\.venv\Scripts\Activate

# Activate (CMD)
.\.venv\Scripts\activate.bat
```

> ⚠️ **Common mistake:** Don't forget the space in `python -m venv .venv`  
> ❌ Wrong: `python -m venv.venv`

---

### 🧰 Step 2 — Install Dependencies

```bash
python -m pip install --upgrade pip

pip install sentence-transformers langchain langchain-groq langchain-community langchain-huggingface einops faiss-cpu
```

---

### 🔑 Step 3 — Set API Keys

You need two API keys:

1. **Groq API Key** — For using the `ChatGroq` LLM.  
   👉 Create one for free at: [https://console.groq.com/keys](https://console.groq.com/keys)

2. **HuggingFace API Token** — For downloading embedding models.  
   👉 Generate your token here: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

#### Option A: Add them manually via environment variables

```bash
setx GROQ_API_KEY "your_groq_api_key_here"
setx HF_TOKEN "your_huggingface_token_here"
```

#### Option B: Store them in a local file (`local_secrets.py`)

Create a file named `local_secrets.py` inside your project folder:

```python
# local_secrets.py
GROQ_API_KEY = "your_groq_api_key_here"
HF_TOKEN = "your_huggingface_token_here"
```

This method keeps your credentials safe from version control.

---

### 📄 Step 4 — Add PDFs

Place your PDFs in the `documents/` folder:

```
RAG/
└── documents/
    ├── ECAPA-TDNN.pdf
    ├── SKA-TDNN.pdf
    ├── InfoEdge_Data_Scientist_2025.pdf
    └── YourPaper.pdf
```

---

### 🚀 Step 5 — Run the Script

```bash
python rag_basics.py
```

Expected output:

```
Loaded 4 PDF files
Created 152 text chunks
<AI-generated answer summarizing your PDFs>
```

---

## 🧠 Code Explanation

### 1️⃣ PDF Loading

```python
pdf_files = glob("./documents/*.pdf")
for file in pdf_files:
    loader = PyPDFLoader(file)
    documents.extend(loader.load())
```
Loads all PDFs from the `documents/` folder.

---

### 2️⃣ Text Chunking

```python
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
texts = text_splitter.split_documents(documents)
```
Splits large texts into small overlapping chunks to maintain context during embedding.

---

### 3️⃣ Embeddings

```python
embedding = HuggingFaceEmbeddings(model_name="nomic-ai/nomic-embed-text-v1.5")
```
Converts each text chunk into a high-dimensional vector using a pre-trained model.

---

### 4️⃣ FAISS Vector Store

```python
retriever = FAISS.from_documents(texts, embedding).as_retriever(search_kwargs={"k": 20})
```
Stores embeddings in a FAISS database for efficient vector similarity search.

---

### 5️⃣ LLM and RAG Chain

```python
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.9)
chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
```

The RetrievalQA chain combines:
- **Retriever:** Fetches top relevant chunks.
- **LLM:** Generates final answers grounded in retrieved content.

---

### 6️⃣ Query and Output

```python
query = "Give me summary of this?"
result = chain.run(query)
print(result)
```
Sends the question to the RAG system and prints the LLM’s contextual answer.

---

## 💬 Example Queries

```text
"Summarize ECAPA-TDNN paper."
"Explain InfoEdge Data Scientist report."
"Compare SKA-TDNN and ECAPA-TDNN models."
"List all datasets mentioned across papers."
```

---

## 🧩 Common Issues

| Issue | Cause | Fix |
|-------|--------|-----|
| `No module named faiss` | Missing FAISS | Install `faiss-cpu` |
| `Invalid API key` | Missing Groq/HF keys | Add them to `local_secrets.py` |
| `No PDF files found` | Wrong folder path | Ensure PDFs in `/documents` |

---

## 🧾 Output Example

```
Loaded 5 PDF files
Created 215 text chunks
Answer:
The ECAPA-TDNN architecture enhances speaker verification by integrating channel attention and residual layers...
```

---

## 🧱 Future Enhancements

- Gradio/Streamlit UI for interactive Q&A  
- PDF upload via web interface  
- Support for YouTube transcripts or DOCX files  
- Local persistence of FAISS index for faster reloads  

---

## 🧑‍💻 Author

Developed by **Prince Verma**  
RAG System using LangChain + HuggingFace + Groq

---

## 📜 License

This project is for **academic and educational purposes** only.

---
