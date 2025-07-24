import streamlit as st
import os
import io
import PyPDF2 # Import PyPDF2 for PDF handling
import json
import requests # Still useful for direct API calls to Gemini for structured output
import google.generativeai as genai # Direct Gemini SDK
import numpy as np # For numerical operations with embeddings
import faiss # For efficient similarity search

# --- Configuration ---
# Access the API key securely from Streamlit's secrets
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=GEMINI_API_KEY)
except KeyError:
    st.error("Gemini API key not found. Please set it in Streamlit secrets.")
    st.stop() # Stop the app if API key is missing

# Initialize Gemini models directly
CHAT_MODEL = genai.GenerativeModel('gemini-pro')
EMBEDDING_MODEL = genai.GenerativeModel('embedding-001')

# --- Session State Initialization ---
if "chat_display_history" not in st.session_state:
    st.session_state.chat_display_history = []
if "document_chunks" not in st.session_state:
    st.session_state.document_chunks = [] # Stores original text chunks
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
if 'summary_data' not in st.session_state:
    st.session_state['summary_data'] = None

# --- Helper Functions for Document Processing (RAG) ---

def get_text_chunks(text, chunk_size=1000, chunk_overlap=200):
    """Simple text chunking based on character count."""
    chunks = []
    current_chunk = ""
    words = text.split() # Simple word split
    for word in words:
        if len(current_chunk) + len(word) + 1 <= chunk_size: # +1 for space
            current_chunk += (word + " ").strip()
        else:
            chunks.append(current_chunk)
            current_chunk = word + " "
    if current_chunk:
        chunks.append(current_chunk)
    
    # A more sophisticated chunking would handle overlap properly.
    # For a hackathon, this simple split is often sufficient.
    return chunks

def get_embeddings(texts):
    """Generates embeddings for a list of texts using Gemini Embedding model."""
    embeddings = []
    for text in texts:
        try:
            # The genai.embed_content returns a dict with 'embedding' key
            response = EMBEDDING_MODEL.embed_content(model="models/embedding-001", content=text)
            embeddings.append(response['embedding'])
        except Exception as e:
            st.error(f"Error generating embedding for text: {e}")
            embeddings.append(None) # Append None to maintain list length
    return [e for e in embeddings if e is not None] # Filter out failures

def process_document_for_rag(uploaded_file):
    """
    Extracts text, chunks it, generates embeddings, and creates a FAISS index.
    """
    with st.spinner("Processing document for Q&A..."):
        file_content = None
        if uploaded_file.type == "text/plain":
            file_content = io.StringIO(uploaded_file.getvalue().decode("utf-8")).read()
        elif uploaded_file.type == "application/pdf":
            file_content = extract_text_from_pdf(uploaded_file)
        else:
            st.error("Unsupported file type for processing.")
            return

        if not file_content:
            st.error("Could not extract content from the uploaded document.")
            return

        st.info("Text extracted. Chunking and embedding...")
        chunks = get_text_chunks(file_content)
        st.session_state.document_chunks = chunks # Store original chunks

        embeddings_list = get_embeddings(chunks)
        if not embeddings_list:
            st.error("Failed to generate any embeddings from the document.")
            return

        # Convert to numpy array
        embeddings_np = np.array(embeddings_list).astype('float32')

        # Create a FAISS index
        dimension = embeddings_np.shape[1]
        faiss_index = faiss.IndexFlatL2(dimension) # L2 distance for similarity
        faiss_index.add(embeddings_np)
        
        st.session_state.faiss_index = faiss_index
        st.success("Document processed and ready for questions!")
        
        # Clear chat history when new document is loaded
        st.session_state.chat_display_history = []
        # No LangChain memory to clear directly here, just the display history

# Function to call the Gemini API for summarization (from your original code, adapted)
def summarize_document_initial(document_text: str):
    """
    Summarizes the provided document text using the Gemini 2.0 Flash model,
    extracting coverages, exclusions, and policy details in a structured JSON format.
    """
    st.info("Analyzing document and generating summary...")

    prompt = f"""
    You are an AI assistant specialized in summarizing insurance documents.
    Please read the following insurance document and extract the following information in a structured JSON format:
    1.  A concise overall 'summary' of the document.
    2.  A list of 'coverages' provided by the policy.
    3.  A list of 'exclusions' (what is not covered) by the policy.
    4.  'policyDetails' as an object containing:
        -   'policyNumber' (if found)
        -   'policyHolder' (if found)
        -   'effectiveDate' (if found, e.g., "YYYY-MM-DD")
        -   'expirationDate' (if found, e.g., "YYYY-MM-DD")
        -   'premium' (if found, e.g., "USD 1200" or "1200 per year")
        -   'otherDetails' (a list of any other significant policy details not covered above).

    If a piece of information is not explicitly found, use "N/A" for strings or an empty list for arrays.
    Ensure the output is valid JSON.

    Document:
    ---
    {document_text}
    ---
    """
    
    # Use the CHAT_MODEL for the summarization task
    # We still use requests directly for precise JSON schema control, as genai.GenerativeModel 
    # might not directly support response_schema in the same way as the raw API payload.
    # Alternatively, you could just instruct the model to output JSON and then parse it.

    # Simpler approach: Just ask the model for JSON and parse it
    try:
        response = CHAT_MODEL.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                response_mime_type="application/json"
            )
        )
        json_string = response.text
        if json_string.startswith("```json") and json_string.endswith("```"):
            json_string = json_string[7:-3].strip()
        return json.loads(json_string)
    except Exception as e:
        st.error(f"Error during summarization: {e}")
        st.json(response.to_dict() if 'response' in locals() else "No response object") # For debugging
        return None


# Function to extract text from PDF files
def extract_text_from_pdf(uploaded_file):
    """
    Extracts text content from an uploaded PDF file.
    """
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page_text = pdf_reader.pages[page_num].extract_text()
            if page_text: # Ensure text is not None or empty
                text += page_text + "\n" # Add newline for better readability between pages
        return text
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return None

# --- Streamlit UI ---
st.set_page_config(page_title="Insurance Document AI Assistant", layout="centered")

st.markdown(
    """
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
    }
    h1 {
        color: #1e3a8a; /* Dark blue */
        text-align: center;
        margin-bottom: 30px;
    }
    .stFileUploader > div > button {
        background-color: #4CAF50; /* Green */
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
        border: none;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }
    .stFileUploader > div > button:hover {
        background-color: #45a049;
    }
    .stButton > button {
        background-color: #1e3a8a; /* Dark blue */
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
        border: none;
        cursor: pointer;
        transition: background-color 0.3s ease;
        display: block;
        margin: 20px auto;
    }
    .stButton > button:hover {
        background-color: #15306b;
    }
    .summary-section {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        margin-top: 30px;
    }
    .summary-section h3 {
        color: #1e3a8a;
        border-bottom: 2px solid #e0e0e0;
        padding-bottom: 10px;
        margin-bottom: 15px;
    }
    .summary-section ul {
        list-style-type: disc;
        margin-left: 20px;
        padding-left: 0;
    }
    .summary-section li {
        margin-bottom: 8px;
        color: #333;
    }
    .policy-details-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 15px;
    }
    .policy-details-table th, .policy-details-table td {
        border: 1px solid #ddd;
        padding: 8px;
        text-align: left;
    }
    .policy-details-table th {
        background-color: #f2f2f2;
        color: #1e3a8a;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📄 Insurance Document AI Assistant")
st.write("Upload your insurance document (text file or PDF) to get a summary and ask questions.")

# --- Sidebar for Document Upload and Initial Summary ---
with st.sidebar:
    st.header("Document Operations")
    uploaded_file_sidebar = st.file_uploader("Choose a document (.txt or .pdf)", type=["txt", "pdf"], key="sidebar_uploader")

    if uploaded_file_sidebar is not None:
        file_content_sidebar = None
        # Read the file content for initial display and subsequent processing
        if uploaded_file_sidebar.type == "text/plain":
            file_content_sidebar = io.StringIO(uploaded_file_sidebar.getvalue().decode("utf-8")).read()
        elif uploaded_file_sidebar.type == "application/pdf":
            file_content_sidebar = extract_text_from_pdf(uploaded_file_sidebar)
        else:
            st.error("Unsupported file type. Please upload a .txt or .pdf file.")

        if file_content_sidebar:
            st.subheader("Document Preview:")
            with st.expander("View Document Content"):
                st.text_area("Content", file_content_sidebar, height=200, disabled=True)

            if st.button("Generate Summary"):
                if file_content_sidebar:
                    with st.spinner("Generating initial summary..."):
                        summary_data = summarize_document_initial(file_content_sidebar)
                        st.session_state['summary_data'] = summary_data
                        if summary_data:
                            st.success("Initial summary generated!")
                        else:
                            st.error("Failed to generate initial summary.")
                else:
                    st.warning("Could not extract content from the uploaded document.")

            if st.button("Enable Q&A on Document"):
                if file_content_sidebar:
                    process_document_for_rag(uploaded_file_sidebar)
                else:
                    st.warning("Could not extract content from the uploaded document to enable Q&A.")
    else:
        st.info("Upload a .txt or .pdf file to begin.")


# --- Main Content Area ---

# Display initial summary if available
if st.session_state.get('summary_data'):
    st.markdown('<div class="summary-section">', unsafe_allow_html=True)
    st.subheader("Initial Summary Report")

    st.markdown("### Overall Summary")
    st.write(st.session_state['summary_data'].get("summary", "N/A"))

    st.markdown("### Coverages")
    if st.session_state['summary_data'].get("coverages"):
        for coverage in st.session_state['summary_data']["coverages"]:
            st.markdown(f"- {coverage}")
    else:
        st.write("No specific coverages found or identified.")

    st.markdown("### Exclusions")
    if st.session_state['summary_data'].get("exclusions"):
        for exclusion in st.session_state['summary_data']["exclusions"]:
            st.markdown(f"- {exclusion}")
    else:
        st.write("No specific exclusions found or identified.")

    st.markdown("### Policy Details")
    policy_details = st.session_state['summary_data'].get("policyDetails", {})
    if policy_details:
        st.markdown(f"""
        <table class="policy-details-table">
            <tr><th>Detail</th><th>Value</th></tr>
            <tr><td>Policy Number</td><td>{policy_details.get("policyNumber", "N/A")}</td></tr>
            <tr><td>Policy Holder</td><td>{policy_details.get("policyHolder", "N/A")}</td></tr>
            <tr><td>Effective Date</td><td>{policy_details.get("effectiveDate", "N/A")}</td></tr>
            <tr><td>Expiration Date</td><td>{policy_details.get("expirationDate", "N/A")}</td></tr>
            <tr><td>Premium</td><td>{policy_details.get("premium", "N/A")}</td></tr>
        </table>
        """, unsafe_allow_html=True)

        if policy_details.get("otherDetails"):
            st.markdown("#### Other Policy Details")
            for detail in policy_details["otherDetails"]:
                st.markdown(f"- {detail}")
    else:
        st.write("No specific policy details found or identified.")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---") # Separator between summary and chat

# --- Chat Interface ---
st.subheader("Chat with the AI Assistant")

# Display chat messages from history
for message in st.session_state.chat_display_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask a question about the document..."):
    # Add user message to chat display history
    st.session_state.chat_display_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            ai_response = "I'm sorry, I cannot answer questions about a document until one is processed for Q&A. Please upload a document and click 'Enable Q&A on Document' in the sidebar."

            if st.session_state.faiss_index and st.session_state.document_chunks:
                try:
                    # 1. Embed the user's query
                    query_embedding_response = EMBEDDING_MODEL.embed_content(model="models/embedding-001", content=prompt)
                    query_embedding = np.array(query_embedding_response['embedding']).astype('float32').reshape(1, -1)

                    # 2. Search the FAISS index for relevant chunks
                    D, I = st.session_state.faiss_index.search(query_embedding, k=3) # k=3 for top 3 results
                    
                    retrieved_chunks_content = []
                    source_info = []
                    for idx in I[0]:
                        if idx < len(st.session_state.document_chunks):
                            chunk_content = st.session_state.document_chunks[idx]
                            retrieved_chunks_content.append(chunk_content)
                            source_info.append(f"Chunk Index {idx}") # Simple source info

                    context = "\n\n".join(retrieved_chunks_content)

                    # 3. Build the prompt with RAG context and chat history
                    # Prepare chat history for Gemini API (assuming simple 'user'/'model' roles)
                    gemini_chat_history = []
                    # Keep a window of recent history for LLM context, adapt as needed
                    history_window_size = 5 # last 5 exchanges
                    
                    # Convert display history to Gemini API format and add to gemini_chat_history
                    for msg in st.session_state.chat_display_history[-history_window_size:]:
                        if msg["role"] == "user":
                            gemini_chat_history.append({"role": "user", "parts": [{"text": msg["content"]}]})
                        elif msg["role"] == "assistant":
                            gemini_chat_history.append({"role": "model", "parts": [{"text": msg["content"]}]})
                    
                    # Ensure the last message in gemini_chat_history is from the user
                    if gemini_chat_history and gemini_chat_history[-1]["role"] != "user":
                         # This should ideally not happen if 'prompt' is always the latest user input
                         pass # handle if needed
                    else: # Add the current user prompt if it's not already the last one
                         gemini_chat_history.append({"role": "user", "parts": [{"text": prompt}]})


                    # The actual prompt for the LLM
                    rag_prompt = f"""
                    You are an AI assistant. Answer the user's question based ONLY on the provided document context.
                    If the answer is not found in the context, state that you cannot find the information.

                    Document Context:
                    ---
                    {context}
                    ---

                    User Question: {prompt}
                    """
                    
                    # Create a chat session to maintain conversation context (if using chat models)
                    # For a single request with full history, you can just pass 'contents' directly
                    
                    # Build the contents list for the API call
                    final_contents = []
                    
                    # Append history up to the current user prompt
                    for msg in st.session_state.chat_display_history:
                        if msg["role"] == "user":
                            final_contents.append({"role": "user", "parts": [{"text": msg["content"]}]})
                        elif msg["role"] == "assistant":
                            final_contents.append({"role": "model", "parts": [{"text": msg["content"]}]})

                    # Replace the *last* user message (which is 'prompt') with the RAG-augmented prompt
                    # Or, more robustly, append a new user message containing the RAG prompt
                    
                    # This is simpler and avoids complex history manipulation:
                    # Treat the RAG prompt as the *current* user input to the model for this turn
                    
                    # The generate_content method for the model takes a list of Content objects
                    # For a simple turn, we can send just the rag_prompt.
                    # For a chat session with history, we typically use chat.send_message.
                    # Let's adjust to use chat.send_message for proper multi-turn context
                    
                    chat_session = CHAT_MODEL.start_chat(history=gemini_chat_history[:-1]) # Start with history excluding current prompt
                    response = chat_session.send_message(rag_prompt)
                    ai_response = response.text
                    
                    if source_info:
                        ai_response += "\n\n---\n**Sources from Document:**\n" + "\n".join(source_info)

                except Exception as e:
                    st.error(f"Error during Q&A: {e}")
                    ai_response = "An error occurred while trying to answer your question from the document. Please try again."
            else:
                # If no document is processed for RAG, allow general chat
                try:
                    # Prepare chat history for Gemini API for general chat
                    gemini_chat_history = []
                    history_window_size = 5
                    for msg in st.session_state.chat_display_history[-history_window_size:]:
                        if msg["role"] == "user":
                            gemini_chat_history.append({"role": "user", "parts": [{"text": msg["content"]}]})
                        elif msg["role"] == "assistant":
                            gemini_chat_history.append({"role": "model", "parts": [{"text": msg["content"]}]})
                    
                    chat_session = CHAT_MODEL.start_chat(history=gemini_chat_history[:-1]) # Exclude current prompt
                    response = chat_session.send_message(prompt)
                    ai_response = response.text

                except Exception as e:
                    st.error(f"Error during general chat: {e}")
                    ai_response = "An error occurred during general conversation. Please try again."

            st.markdown(ai_response)
            # Add AI message to chat display history
            st.session_state.chat_display_history.append({"role": "assistant", "content": ai_response})
