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

# Initialize Gemini models directly with specific names
# 'gemini-1.5-flash' is often the most recommended balance of speed and capability for chat.
CHAT_MODEL = genai.GenerativeModel('gemini-1.5-flash')
# 'models/text-embedding-004' is the latest recommended model for embeddings.
EMBEDDING_MODEL_NAME = "models/text-embedding-004"

# --- Session State Initialization ---
# This stores messages for displaying in the UI
if "chat_display_history" not in st.session_state:
    st.session_state.chat_display_history = []
# This stores original text chunks from the uploaded document for RAG
if "document_chunks" not in st.session_state:
    st.session_state.document_chunks = []
# This stores the FAISS index for efficient similarity search
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
# This stores the summary data from the initial summarization
if 'summary_data' not in st.session_state:
    st.session_state['summary_data'] = None

# --- Helper Functions for Document Processing (RAG) ---

def get_text_chunks(text, chunk_size=1000, chunk_overlap=200):
    """
    Splits text into chunks with a specified overlap.
    This is a basic implementation; for production, consider more sophisticated splitters
    like those from NLTK or SpaCy, or a dedicated text splitter library.
    """
    chunks = []
    if not text:
        return chunks

    # Simple character-based chunking with overlap
    start_idx = 0
    while start_idx < len(text):
        end_idx = start_idx + chunk_size
        chunk = text[start_idx:end_idx]
        chunks.append(chunk)
        
        if end_idx >= len(text):
            break
        
        # Move start_idx back by chunk_overlap for the next chunk
        # Ensure start_idx doesn't go negative
        start_idx += (chunk_size - chunk_overlap)
        start_idx = max(0, start_idx) 
        
    return chunks

def get_embeddings(texts):
    """
    Generates embeddings for a list of texts using the specified Gemini Embedding model.
    Handles potential errors during embedding generation.
    """
    embeddings = []
    for text in texts:
        try:
            # Call embed_content directly on the genai module
            response = genai.embed_content(model=EMBEDDING_MODEL_NAME, content=text)
            embeddings.append(response['embedding'])
        except Exception as e:
            st.error(f"Error generating embedding for text: {e}")
            embeddings.append(None) # Append None to maintain list length for FAISS if needed
    return [e for e in embeddings if e is not None] # Filter out any failed embeddings

def process_document_for_rag(uploaded_file):
    """
    Extracts text from the uploaded document, chunks it, generates embeddings,
    and creates a FAISS index for efficient similarity search.
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
        st.session_state.document_chunks = chunks # Store original chunks for retrieval

        embeddings_list = get_embeddings(chunks)
        if not embeddings_list:
            st.error("Failed to generate any embeddings from the document. Q&A will not be enabled.")
            return

        # Convert embeddings to a NumPy array with float32 type, required by FAISS
        embeddings_np = np.array(embeddings_list).astype('float32')

        # Create a FAISS index: IndexFlatL2 uses L2 (Euclidean) distance for similarity search
        dimension = embeddings_np.shape[1] # Dimension of the embeddings
        faiss_index = faiss.IndexFlatL2(dimension)
        faiss_index.add(embeddings_np) # Add the embeddings to the index
        
        st.session_state.faiss_index = faiss_index
        st.success("Document processed and ready for questions!")
        
        # Clear existing chat history when a new document is loaded to avoid confusion
        st.session_state.chat_display_history = []


# Function to call the Gemini API for initial document summarization
def summarize_document_initial(document_text: str):
    """
    Summarizes the provided document text using the CHAT_MODEL (gemini-1.5-flash),
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
    
    try:
        # Use the CHAT_MODEL to generate content, requesting JSON format
        response = CHAT_MODEL.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                response_mime_type="application/json"
            )
        )
        json_string = response.text
        # Models might wrap JSON in markdown code blocks; remove them if present
        if json_string.startswith("```json") and json_string.endswith("```"):
            json_string = json_string[7:-3].strip()
        return json.loads(json_string)
    except Exception as e:
        st.error(f"Error during summarization: {e}")
        # For debugging, you might want to print the raw response text if available
        # if 'response' in locals() and hasattr(response, 'text'):
        #     st.text(f"Raw model response: {response.text}")
        return None


# Function to extract text from PDF files using PyPDF2
def extract_text_from_pdf(uploaded_file):
    """
    Extracts text content from an uploaded PDF file using PyPDF2.
    Adds a newline character between pages for better readability.
    """
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page_text = pdf_reader.pages[page_num].extract_text()
            if page_text: # Ensure text is not None or empty
                text += page_text + "\n" # Add newline for better separation between pages
        return text
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return None

# --- Streamlit UI Layout and Styling ---
st.set_page_config(page_title="Insurance Document AI Assistant", layout="centered")

# Custom CSS for styling the Streamlit app
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
    /* Style for file uploader button */
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
    /* Style for general Streamlit buttons */
    .stButton > button {
        background-color: #1e3a8a; /* Dark blue */
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
        border: none;
        cursor: pointer;
        transition: background-color 0.3s ease;
        display: block; /* Make buttons block-level for centering */
        margin: 20px auto; /* Center buttons horizontally */
    }
    .stButton > button:hover {
        background-color: #15306b;
    }
    /* Styling for the summary output section */
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
    /* Styling for policy details table */
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

# --- Sidebar for Document Upload and Initial Summary/Q&A Enablement ---
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

            # Button to trigger initial summarization
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

            # Button to enable Q&A (RAG) on the document
            if st.button("Enable Q&A on Document"):
                if file_content_sidebar:
                    process_document_for_rag(uploaded_file_sidebar)
                else:
                    st.warning("Could not extract content from the uploaded document to enable Q&A.")
    else:
        st.info("Upload a .txt or .pdf file to begin.")


# --- Main Content Area: Display Summary (if available) ---

# Display initial summary report if it has been generated
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

# Accept user input for chat
if prompt := st.chat_input("Ask a question about the document..."):
    # Add user message to chat display history
    st.session_state.chat_display_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            ai_response = "I'm sorry, I cannot answer questions about a document until one is processed for Q&A. Please upload a document and click 'Enable Q&A on Document' in the sidebar."

            # Check if RAG is enabled (FAISS index and chunks are available)
            if st.session_state.faiss_index and st.session_state.document_chunks:
                try:
                    # 1. Embed the user's query for similarity search
                    query_embedding_response = genai.embed_content(model=EMBEDDING_MODEL_NAME, content=prompt)
                    query_embedding = np.array(query_embedding_response['embedding']).astype('float32').reshape(1, -1)

                    # 2. Search the FAISS index for the most relevant document chunks
                    D, I = st.session_state.faiss_index.search(query_embedding, k=3) # Retrieve top 3 relevant chunks
                    
                    retrieved_chunks_content = []
                    source_info = []
                    for idx in I[0]:
                        # Ensure the retrieved index is within the bounds of stored chunks
                        if 0 <= idx < len(st.session_state.document_chunks):
                            chunk_content = st.session_state.document_chunks[idx]
                            retrieved_chunks_content.append(chunk_content)
                            source_info.append(f"Chunk Index {idx}") # Simple source info for the user

                    context = "\n\n".join(retrieved_chunks_content) # Combine retrieved chunks into context string

                    # 3. Build the prompt with RAG context and chat history for the LLM
                    # Prepare chat history for Gemini API. Gemini chat history expects alternating user/model roles.
                    cleaned_history_for_gemini = []
                    # Iterate through the display history to build Gemini-compatible history
                    for i in range(len(st.session_state.chat_display_history)):
                        msg = st.session_state.chat_display_history[i]
                        # The *last* message in chat_display_history is the current user prompt,
                        # which will be sent as part of the `send_message` call, not in `history`.
                        if i == len(st.session_state.chat_display_history) - 1 and msg["role"] == "user":
                            break 
                        cleaned_history_for_gemini.append({"role": msg["role"], "parts": [{"text": msg["content"]}]})

                    # The RAG-augmented prompt, instructing the AI to use the context
                    rag_augmented_prompt = f"""
                    You are an AI assistant. Answer the user's question based ONLY on the provided document context.
                    If the answer is not found in the context, state that you cannot find the information.

                    Document Context:
                    ---
                    {context}
                    ---

                    User Question: {prompt}
                    """
                    
                    # Start a chat session with the accumulated history
                    chat_session = CHAT_MODEL.start_chat(history=cleaned_history_for_gemini)
                    
                    # Send the RAG-augmented prompt as the current message in the chat session
                    response = chat_session.send_message(rag_augmented_prompt)
                    ai_response = response.text
                    
                    # Append source information if available
                    if source_info:
                        ai_response += "\n\n---\n**Sources from Document:**\n" + "\n".join(source_info)

                except Exception as e:
                    st.error(f"Error during Q&A: {e}")
                    ai_response = "An error occurred while trying to answer your question from the document. Please try again."
            else:
                # If no document is processed for RAG, allow general chat using the CHAT_MODEL
                try:
                    # Prepare chat history for Gemini API for general chat (same logic as RAG path)
                    cleaned_history_for_gemini = []
                    for i in range(len(st.session_state.chat_display_history)):
                        msg = st.session_state.chat_display_history[i]
                        if i == len(st.session_state.chat_display_history) - 1 and msg["role"] == "user":
                            break
                        cleaned_history_for_gemini.append({"role": msg["role"], "parts": [{"text": msg["content"]}]})
                    
                    chat_session = CHAT_MODEL.start_chat(history=cleaned_history_for_gemini)
                    response = chat_session.send_message(prompt) # Send the original user prompt
                    ai_response = response.text

                except Exception as e:
                    st.error(f"Error during general chat: {e}")
                    ai_response = "An error occurred during general conversation. Please try again."

            # Display the AI's response in the chat interface
            st.markdown(ai_response)
            # Add the AI's response to the display history for future turns
            st.session_state.chat_display_history.append({"role": "assistant", "content": ai_response})
