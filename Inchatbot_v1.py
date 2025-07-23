import streamlit as st
import os
import io
import PyPDF2 # Import PyPDF2 for PDF handling
import json # For structured JSON output from Gemini
import requests # For direct API calls if not using google-generativeai for main summary

# For RAG and Chat History (install these via requirements.txt)
from langchain_google_generativeai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.messages import HumanMessage, AIMessage

# --- Configuration ---
# Access the API key securely from Streamlit's secrets
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except KeyError:
    st.error("Gemini API key not found. Please set it in Streamlit secrets or as an environment variable.")
    st.stop() # Stop the app if API key is missing

# Initialize LLM and Embeddings for LangChain
# Using gemini-pro for chat and embedding-001 for embeddings
llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GEMINI_API_KEY, temperature=0.5) # Adjust temperature for creativity
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)

# --- Session State Initialization ---
# This stores messages for displaying in the UI
if "chat_display_history" not in st.session_state:
    st.session_state.chat_display_history = []
# This stores the LangChain memory object for conversational context
if "langchain_memory" not in st.session_state:
    # k=5 means it will remember the last 5 exchanges (user + AI turns)
    st.session_state.langchain_memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=5)
# This stores the vector store for RAG
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
# This stores the summary data from the initial summarization (from your original code)
if 'summary_data' not in st.session_state:
    st.session_state['summary_data'] = None

# Function to call the Gemini API for summarization (from your original code)
def summarize_document_initial(document_text: str):
    """
    Summarizes the provided document text using the Gemini 2.0 Flash model,
    extracting coverages, exclusions, and policy details in a structured JSON format.
    """
    st.info("Analyzing document and generating summary...")

    chat_history = []
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
    chat_history.append({
        "role": "user",
        "parts": [{"text": prompt}]
    })

    payload = {
        "contents": chat_history,
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "summary": {"type": "STRING"},
                    "coverages": {
                        "type": "ARRAY",
                        "items": {"type": "STRING"}
                    },
                    "exclusions": {
                        "type": "ARRAY",
                        "items": {"type": "STRING"}
                    },
                    "policyDetails": {
                        "type": "OBJECT",
                        "properties": {
                            "policyNumber": {"type": "STRING"},
                            "policyHolder": {"type": "STRING"},
                            "effectiveDate": {"type": "STRING"},
                            "expirationDate": {"type": "STRING"},
                            "premium": {"type": "STRING"},
                            "otherDetails": {
                                "type": "ARRAY",
                                "items": {"type": "STRING"}
                            }
                        },
                        "propertyOrdering": ["policyNumber", "policyHolder", "effectiveDate", "expirationDate", "premium", "otherDetails"]
                    }
                },
                "propertyOrdering": ["summary", "coverages", "exclusions", "policyDetails"]
            }
        }
    }

    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

    try:
        response = requests.post(api_url, headers={'Content-Type': 'application/json'}, json=payload)
        response.raise_for_status() # Raise an exception for HTTP errors
        result = response.json()

        if result.get("candidates") and result["candidates"][0].get("content") and result["candidates"][0]["content"].get("parts"):
            json_string = result["candidates"][0]["content"]["parts"][0]["text"]
            if json_string.startswith("```json") and json_string.endswith("```"):
                json_string = json_string[7:-3].strip()
            return json.loads(json_string)
        else:
            st.error("Error: Could not get a valid response from the summarization model.")
            st.json(result)
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Network or API error: {e}")
        return None
    except json.JSONDecodeError as e:
        st.error(f"Error decoding JSON response from model: {e}")
        st.text(response.text)
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
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
            text += pdf_reader.pages[page_num].extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return None

# Function to process document for RAG
def process_document_for_rag(uploaded_file):
    """
    Loads, chunks, and embeds the document content into a vector store for RAG.
    """
    with st.spinner("Processing document for Q&A..."):
        file_content = None
        if uploaded_file.type == "text/plain":
            file_content = io.StringIO(uploaded_file.getvalue().decode("utf-8")).read()
            # For TextLoader, we need a file-like object or a path.
            # Create a temporary file or use io.StringIO directly as a source
            loader = TextLoader(io.StringIO(file_content))
        elif uploaded_file.type == "application/pdf":
            file_content = extract_text_from_pdf(uploaded_file)
            if not file_content:
                return # Stop if PDF extraction failed
            # For PDFLoader, it usually expects a file path.
            # For in-memory, we can treat the extracted text as a single document.
            # If you need to use PyPDFLoader directly, you'd save uploaded_file to a temp file.
            # For simplicity in a hackathon, we'll use TextLoader with the extracted text.
            loader = TextLoader(io.StringIO(file_content))
        else:
            st.error("Unsupported file type for processing.")
            return

        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)

        # Create or update vector store with new chunks
        st.session_state.vector_store = Chroma.from_documents(chunks, embeddings)
        st.success("Document processed and ready for questions!")
        # Clear chat history when new document is loaded to avoid confusion
        st.session_state.chat_display_history = []
        st.session_state.langchain_memory.clear() # Clear LangChain memory too


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
                    # Pass the original uploaded file object to process_document_for_rag
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

            if st.session_state.vector_store:
                try:
                    # Create a conversational retrieval chain
                    qa_chain = ConversationalRetrievalChain.from_llm(
                        llm=llm,
                        retriever=st.session_state.vector_store.as_retriever(),
                        memory=st.session_state.langchain_memory, # Use the LangChain memory
                        return_source_documents=True # Optional: to show sources
                    )

                    # Invoke the chain with the user's question
                    response = qa_chain.invoke({"question": prompt})
                    ai_response = response["answer"]

                    # Add source documents if available
                    if response.get("source_documents"):
                        ai_response += "\n\n---\n**Sources from Document:**\n"
                        for i, doc in enumerate(response["source_documents"]):
                            # You might want to format this better, e.g., show page numbers if available
                            source_content = doc.page_content.replace('\n', ' ')
                            ai_response += f"- *Source {i+1}:* \"{source_content[:150]}...\"\n" # Show first 150 chars

                except Exception as e:
                    st.error(f"Error during Q&A: {e}")
                    ai_response = "An error occurred while trying to answer your question. Please try again."
            else:
                # If no document is processed for RAG, still allow general chat but without document context
                # The LangChain memory will still work for general conversation
                try:
                    # Manually add current prompt to memory for a direct LLM call
                    # LangChain's memory automatically adds to its buffer when used in a chain.
                    # For a direct call, we need to manually pass the history and update it.
                    
                    # Get current chat history from LangChain memory
                    current_langchain_history = st.session_state.langchain_memory.buffer_as_messages
                    
                    # Prepare messages for the LLM call
                    messages_for_llm = current_langchain_history + [HumanMessage(content=prompt)]
                    
                    direct_llm_response = llm.invoke(messages_for_llm)
                    ai_response = direct_llm_response.content
                    
                    # Manually save the context of this general conversation to LangChain memory
                    st.session_state.langchain_memory.save_context({"input": prompt}, {"output": ai_response})

                except Exception as e:
                    st.error(f"Error during general chat: {e}")
                    ai_response = "An error occurred during general conversation. Please try again."

            st.markdown(ai_response)
            # Add AI message to chat display history
            st.session_state.chat_display_history.append({"role": "assistant", "content": ai_response})