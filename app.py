import streamlit as st
import requests
import pymupdf



from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "gemma3:1b"


# ------------------------------------------------
# Ollama
# ------------------------------------------------

def ask_ollama(messages):

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "messages": messages,
            "stream": False
        }
    )

    response.raise_for_status()

    return response.json()["message"]["content"]


# ------------------------------------------------
# PDF extraction
# ------------------------------------------------

def extract_pdf(uploaded_file):

    pdf_bytes = uploaded_file.read()

    document = pymupdf.open(stream=pdf_bytes, filetype="pdf")

    text = ""

    for page in document:
        text += page.get_text()

    return text


# ------------------------------------------------
# Chunking
# ------------------------------------------------

def create_chunks(text, chunk_size=800, overlap=100):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# ------------------------------------------------
# Simple Retrieval
# ------------------------------------------------

def retrieve_chunks(question, chunks, k=3):

    if not chunks:
        return []

    vectorizer = TfidfVectorizer()

    vectors = vectorizer.fit_transform(
        chunks + [question]
    )

    chunk_vectors = vectors[:-1]
    question_vector = vectors[-1]

    similarities = cosine_similarity(
        question_vector,
        chunk_vectors
    )[0]

    best_indices = similarities.argsort()[-k:][::-1]

    return [chunks[i] for i in best_indices]


# ------------------------------------------------
# Streamlit UI
# ------------------------------------------------

st.set_page_config(
    page_title="Local AI Assistant",
    page_icon="🤖"
)

st.title("🤖 Local AI Assistant")

st.caption("Powered by Gemma 3 1B + Ollama")


# ------------------------------------------------
# Session memory
# ------------------------------------------------

if "messages" not in st.session_state:

    st.session_state.messages = []


if "chunks" not in st.session_state:

    st.session_state.chunks = []


# ------------------------------------------------
# Sidebar
# ------------------------------------------------

with st.sidebar:

    st.header("📄 Documents")

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"]
    )

    if uploaded_file:

        if st.button("Process PDF"):

            with st.spinner("Reading PDF..."):

                text = extract_pdf(uploaded_file)

                st.session_state.chunks = create_chunks(text)

            st.success(
                f"PDF processed: "
                f"{len(st.session_state.chunks)} chunks"
            )


# ------------------------------------------------
# Show conversation
# ------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ------------------------------------------------
# User input
# ------------------------------------------------

user_input = st.chat_input(
    "Ask me something..."
)


if user_input:

    # show user
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    with st.chat_message("user"):

        st.markdown(user_input)


    # ------------------------------------------------
    # Retrieval
    # ------------------------------------------------

    context = ""

    if st.session_state.chunks:

        retrieved = retrieve_chunks(
            user_input,
            st.session_state.chunks,
            k=3
        )

        context = "\n\n".join(retrieved)


    # ------------------------------------------------
    # System Prompt
    # ------------------------------------------------

    system_prompt = """
You are a helpful AI assistant.

If document context is provided, use it to answer the user's question.

If the answer cannot be found in the provided document context,
say that you could not find the answer in the document.

Keep answers concise and clear.
"""


    if context:

        system_prompt += f"""

DOCUMENT CONTEXT:

{context}

"""


    messages = [
        {
            "role": "system",
            "content": system_prompt
        }
    ]


    # keep recent conversation
    messages += st.session_state.messages[-8:]


    # ------------------------------------------------
    # Ollama answer
    # ------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:

                answer = ask_ollama(messages)

            except Exception as e:

                answer = f"Ollama error: {e}"

        st.markdown(answer)


    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )