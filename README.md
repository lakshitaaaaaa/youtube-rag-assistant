# youtube-rag-assistant

This project implements a Retrieval-Augmented Generation (RAG) chatbot that answers user questions based on the content of a YouTube video. The system extracts the video transcript, converts it into vector embeddings, retrieves relevant segments, and generates accurate, context-grounded responses using a large language model.

---

## Overview

The purpose of this system is to enable precise and context-grounded question-answering from YouTube videos. The workflow includes:

1. Extracting the transcript of a video  
2. Preprocessing the transcript and dividing it into meaningful chunks  
3. Creating embeddings using a HuggingFace transformer model  
4. Storing embeddings in a FAISS vector store  
5. Retrieving relevant segments for a user query  
6. Constructing a prompt containing retrieved context  
7. Generating an answer from an LLM based solely on the retrieved content  

This architecture ensures factual answers tied directly to the video transcript.

---

## Architecture Diagram (ASCII)

```

+------------------+       +------------------------+      +----------------------+
|  YouTube Video   | --->  | Transcript Fetcher     | ---> | Transcript Cache     |
+------------------+       +------------------------+      +----------------------+
|
v
+------------------------+
| Text Preprocessing     |
| (Chunking & Cleaning)  |
+------------------------+
|
v
+----------------------------------+
| HuggingFace Embeddings           |
+----------------------------------+
|
v
+------------------------------+
| FAISS Vector Store           |
+------------------------------+
|
v
+---------------------------------------+
| Retriever (Similarity Search, k=4)    |
+---------------------------------------+
|
v
+-------------------------------+
| Prompt + Context Assembly     |
+-------------------------------+
|
v
+----------------------+
| LLM (Llama 3)        |
+----------------------+
|
v
+----------------------+
| Final Answer Output  |
+----------------------+

```

---

## Features

- Automatic YouTube transcript extraction (with caching)  
- Recursive text chunking for improved retrieval  
- Embedding generation using `sentence-transformers/all-MiniLM-L6-v2`  
- FAISS vector database for efficient similarity search  
- Clean and modular RAG pipeline using LangChain runnables  
- Strict grounding to transcript context  
- Support for any English-transcript YouTube video  

---

## Project Structure

```
youtube-rag-assistant/
│
├── transcript_cache/            # Stores cached transcripts for reuse
├── youtube-rag-assistant.ipynb  # Main development notebook
├── requirements.txt             # Dependencies
├── README.md                    # Documentation
└── .env                         # Environment variables (ignored in Git)
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

### 2. Install dependencies

Since working directly in Colab, the required libraries are installed within the notebook.

Install core dependencies:

```

pip install -q youtube-transcript-api langchain-community langchain-openai 
faiss-cpu tiktoken python-dotenv

```

Install embedding dependencies:

```

pip install -U sentence-transformers langchain-huggingface numpy

```

### Notes on Dependencies
- HuggingFace sentence-transformers require **PyTorch** internally.  
- Embeddings output PyTorch tensors which are converted to NumPy arrays.  
- FAISS requires NumPy arrays for similarity search.  
- Even if NumPy is not imported in your code, it is required internally.


### 3. Environment Variables

Create a `.env` file in the project root:

```

OPENAI_API_KEY="your_api_key_here"
This key is required because the project uses:

```python
llm = ChatOpenAI(model="llama3", temperature=0.2)
````

---

## Usage

You can run the full workflow inside the uploaded Colab notebook:

1. Open the notebook
2. Install dependencies inside Colab
3. Provide a YouTube video link
4. Ask questions related to the video
5. The system retrieves relevant transcript chunks and generates an answer

---

## How the System Works (Detailed RAG Pipeline)

### 1. Transcript Fetching and Caching

* The system checks the `transcript_cache` directory.
* If a transcript exists, it is loaded.
* Otherwise, it is fetched from YouTube via `YouTubeTranscriptApi` and cached.

### 2. Transcript Preprocessing

The transcript segments are joined into a single text string.
A RecursiveCharacterTextSplitter is applied with:

```
chunk_size = 1000  
chunk_overlap = 200
```

This ensures semantically meaningful splits.

### 3. Embedding Generation

Embeddings are produced using:

```
sentence-transformers/all-MiniLM-L6-v2
```

This model is efficient and well-suited for semantic search tasks.

### 4. Vector Store Creation

Chunks and embeddings are stored in a FAISS index:

```
vector_store = FAISS.from_documents(chunks, embedding_model)
```

### 5. Retrieval

Top-k (k=4) most relevant chunks are retrieved for user questions:

```
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
```

### 6. Prompt Construction

LangChain runnables are used to assemble the final RAG chain:

* RunnableParallel:

  * Retrieves context from FAISS
  * Passes the original question through unchanged

* Prompt:

  * Injects both into a structured prompt template

* LLM:

  * Generates an answer grounded only in transcript content

* Parser:

  * Extracts the final clean text response

### 7. Final Answer

The answer is produced using:

```
main_chain.invoke("Your question here")
```

---

## RAG Chain Architecture

```
User Input Question
           |
           v
+-------------------------+
| RunnableParallel        |
|  context <- retriever   |
|  question <- passthrough|
+-------------------------+
           |
           v
+-------------------------+
| Prompt Template         |
+-------------------------+
           |
           v
+-------------------------+
| LLM (Llama 3)           |
+-------------------------+
           |
           v
+-------------------------+
| Output Parser           |
+-------------------------+
           |
           v
Final Answer
```

---

## Usage

After configuring everything, run:

```python
main_chain.invoke("Can you summarize the video?")
```

The system will:

1. Retrieve most relevant transcript segments
2. Construct a prompt
3. Produce a grounded answer

---

## Limitations

* The system depends on transcript availability.
* Quality of retrieval affects answer accuracy.
* If YouTube disables transcripts, the system cannot process that video.
* The model cannot answer outside the transcript's scope.

---

## Future Improvements

* Add support for multilingual transcripts
* Add web-based UI
* Support for local LLMs
* Improved chunking strategies for long videos
* Store FAISS index persistently

---

## License

This project is for academic use only.

---
