# youtube-rag-assistant

This project implements a Retrieval-Augmented Generation (RAG) chatbot that answers user questions based on the content of a YouTube video. The system extracts the video transcript, converts it into vector embeddings, retrieves relevant segments, and generates accurate, context-grounded responses using a large language model.

---

## Features

* Extracts transcripts from YouTube videos
* Splits transcript into semantically meaningful text chunks
* Embeds text using modern embedding models
* Uses FAISS for fast vector similarity search
* Generates answers grounded in the retrieved context
* Built with LangChain and LLM APIs
* Handles errors such as unavailable or disabled transcripts

---

## Project Structure

```
project/
│
├── youtube-rag-assistant.ipynb   # Colab notebook with complete implementation
├── README.md
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

---

## Usage

You can run the full workflow inside the uploaded Colab notebook:

1. Open the notebook
2. Install dependencies inside Colab
3. Provide a YouTube video link
4. Ask questions related to the video
5. The system retrieves relevant transcript chunks and generates an answer

---

## High-Level Workflow

1. **Transcript Extraction**
   The system retrieves the transcript of the provided YouTube video using `youtube-transcript-api`.

2. **Text Chunking**
   The transcript is split into overlapping chunks for better semantic retrieval.

3. **Embedding Generation**
   Text chunks are converted into dense vector embeddings.

4. **Vector Storage and Retrieval**
   FAISS is used to store embeddings and retrieve the most relevant chunks based on similarity.

5. **Answer Generation**
   The retrieved context is passed to an LLM to generate a grounded answer.

---

## Requirements

Typical dependencies include:

* python-dotenv
* youtube-transcript-api
* langchain
* langchain-openai or other LLM provider
* langchain-community
* faiss-cpu

---

## Troubleshooting

* **TranscriptUnavailable**: The YouTube video may not have transcripts enabled.
* **ImportError with LangChain**: Ensure all LangChain packages are updated:

```bash
pip install -U langchain langchain-core langchain-openai langchain-community
```

* **Colab to GitHub Save Errors**: Download the notebook locally and upload manually if syncing fails.

---

## License

This project is for academic use only.

---
