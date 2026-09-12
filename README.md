# Local AI Assistant with Ollama

A lightweight local AI assistant built with **Ollama**, **Gemma 3 1B**, and **Streamlit**.

The application allows users to chat with a locally running language model and upload PDF documents for simple document-based question answering. The language model runs locally through Ollama, making the project private, lightweight, and easy to run on a personal computer.

## Features

- Local LLM inference using **Ollama**
- Uses **Gemma 3 1B**
- Interactive chat interface with **Streamlit**
- Conversation history
- PDF upload and text extraction
- Automatic document chunking
- Simple Retrieval-Augmented Generation (RAG)
- TF-IDF based retrieval
- Cosine similarity for finding relevant document chunks
- No external LLM API required

## How It Works

The application supports two main modes.

### 1. Normal Chat

User messages are sent to the locally running Gemma 3 1B model through the Ollama API.

Conversation history is stored with Streamlit session state so that recent messages can be used as context.

### 2. PDF Question Answering

When a PDF is uploaded:

1. Text is extracted from the PDF.
2. The extracted text is split into overlapping chunks.
3. TF-IDF representations are created for the chunks and the user query.
4. Cosine similarity is used to find the most relevant chunks.
5. The top matching chunks are added to the model prompt.
6. Gemma 3 1B generates an answer using the retrieved document context.

The simplified pipeline is:

```text
PDF
 |
 v
Text Extraction
 |
 v
Chunking
 |
 v
TF-IDF
 |
 v
Cosine Similarity
 |
 v
Top Relevant Chunks
 |
 v
Gemma 3 1B via Ollama
 |
 v
Answer
```

## Architecture

```text
                  User
                    |
                    v
             Streamlit UI
                    |
          +---------+---------+
          |                   |
          v                   v
   Conversation            Uploaded PDF
      History                  |
          |                Text Extraction
          |                   |
          |                Chunking
          |                   |
          |                Retrieval
          |                   |
          +---------+---------+
                    |
                    v
               Gemma 3 1B
                 Ollama
                    |
                    v
                 Answer
```

## Technologies

- Python
- Ollama
- Gemma 3 1B
- Streamlit
- PyMuPDF
- Scikit-learn
- Requests

## Project Structure

```text
simple_local_ai/
|
|-- app.py
|-- requirements.txt
`-- README.md
```

## Requirements

Before running the project, install:

- Python 3.10+
- Ollama

Ollama can be downloaded from:

https://ollama.com/

## Install the Model

Download Gemma 3 1B:

```bash
ollama pull gemma3:1b
```

Check that the model is installed:

```bash
ollama list
```

You should see:

```text
gemma3:1b
```

You can test the model directly with:

```bash
ollama run gemma3:1b
```

## Installation

Clone the repository:

```bash
git clone <YOUR-REPOSITORY-URL>
cd simple_local_ai
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

A minimal `requirements.txt` can contain:

```text
streamlit
requests
pymupdf
scikit-learn
```

## Running the Application

Make sure Ollama is running, then start the Streamlit application:

```bash
streamlit run app.py
```

If you have multiple Python installations on Windows, you can also use:

```bash
python -m streamlit run app.py
```

The application will usually be available at:

```text
http://localhost:8501
```

## Usage

### Normal Conversation

Type a message such as:

```text
Explain what deep learning is.
```

The request will be processed locally by Gemma 3 1B.

### Chat with a PDF

Upload a PDF using the sidebar. After the document is processed, ask questions such as:

```text
Summarize this document.
```

```text
What is the main contribution of this paper?
```

```text
What dataset was used?
```

```text
What is the educational background described in this CV?
```

The system retrieves relevant text from the document and includes it in the prompt sent to the local model.

## Retrieval Method

The current version intentionally uses a simple retrieval approach.

Document chunks and the user query are transformed with:

```python
TfidfVectorizer
```

Similarity is calculated using:

```python
cosine_similarity
```

The most relevant chunks are then passed to the language model as document context.

This keeps the implementation lightweight and easy to understand without requiring a vector database or a separate embedding model.

## Local Execution

The main language model runs locally:

```text
Ollama
  |
  v
Gemma 3 1B
  |
  v
Local Machine
```

Therefore, normal LLM inference does not require an external API such as OpenAI, Gemini, or Anthropic.

## Current Limitations

This is intentionally a simple first version.

- TF-IDF retrieval instead of semantic embeddings
- Single-PDF workflow
- No OCR for scanned PDFs
- No vector database
- No web search
- No function calling
- No long-term memory
- No source citations in generated answers
- Retrieval quality depends on extracted PDF text
- Gemma 3 1B may be less accurate than larger language models

## Future Improvements

Possible extensions include:

- Ollama-based embedding models
- FAISS vector search
- Semantic retrieval
- Multiple PDF support
- Source and page citations
- Web search
- Function calling
- Calculator tool
- Long-term memory
- Conversation summarization
- Better context management
- RAG evaluation
- Hallucination detection
- Agent routing
- 20 Questions game mode

## Educational Purpose

This project is designed as a simple introduction to building applications with local Large Language Models.

It demonstrates:

- Local LLM inference
- Chat history
- Prompt construction
- PDF processing
- Information retrieval
- Retrieval-Augmented Generation
- Local AI deployment

## License

This project is intended for educational and research purposes.
