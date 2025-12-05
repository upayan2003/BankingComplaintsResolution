import os
import shutil
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

# --- CONFIGURATION ---
PERSIST_DIR = "./banking_chroma_db"

def reset_knowledge_base():
    """
    Clears the Streamlit cache and deletes the persistent vector DB folder.
    This forces a reload/rebuild of the knowledge base on the next run.
    """
    # 1. Clear the function cache so get_banking_retriever runs again
    st.cache_resource.clear()
    
    # 2. Delete the physical folder if it exists
    if os.path.exists(PERSIST_DIR):
        try:
            shutil.rmtree(PERSIST_DIR)
            st.toast("✅ Persistent Database deleted. Reverting to Sample Data.", icon="🗑️")
        except Exception as e:
            st.error(f"Error deleting DB: {e}")
    else:
        st.toast("ℹ️ No persistent Database found. Cache cleared.", icon="🔄")

# --- IN-SCRIPT KNOWLEDGE BASE (FALLBACK) ---
SAMPLE_KB_DATA = [
    {
        "issue": "Information belongs to someone else (LABEL_0)",
        "content": "Complaint: There is a mortgage account on my credit report that belongs to my twin brother. We have similar names but different SSNs. \nResolution: This appears to be a 'mixed file' error. Under the FCRA, credit bureaus must ensure maximum possible accuracy. Action: Verify the consumer's personal identifiers (SSN, DOB). Separate the credit files immediately and send a confirmation of the correction."
    },
    {
        "issue": "Reporting company used your report improperly (LABEL_1)",
        "content": "Complaint: A car dealership pulled my credit report yesterday, but I never visited them or applied for a loan. \nResolution: Accessing a consumer report without 'permissible purpose' violates the FCRA. Action: Investigation required. If the dealership cannot prove the consumer applied for credit, the hard inquiry must be removed/suppressed from the report."
    },
    {
        "issue": "Their investigation did not fix an error on your report (LABEL_2)",
        "content": "Complaint: I disputed a late payment charge last month. You said it was verified, but I have a bank statement proving I paid on time. \nResolution: If a consumer provides new relevant information, the furnisher must conduct a reasonable reinvestigation. Action: Review the proof of payment provided. If valid, update the trade line to 'Current/Paid as Agreed' and notify all bureaus."
    },
    {
        "issue": "Account information incorrect (LABEL_3)",
        "content": "Complaint: My credit card balance is showing as $5,000 on my report, but I paid it down to zero two weeks ago. \nResolution: Data furnishing issues often occur due to reporting cycles. However, furnishers must report accurate information. Action: Check the 'Date Reported'. If the payment was made after that date, explain the cycle. If the report is outdated, trigger an off-cycle update (AUD) to correct the balance."
    },
    {
        "issue": "Account status incorrect (LABEL_4)",
        "content": "Complaint: My closed auto loan is marked as 'Voluntary Surrender' but I paid it off in full. \nResolution: Incorrect status codes can severely damage credit scores. Action: Audit the account history. If paid in full, update the account status code to '13' (Paid or closed/zero balance) or the appropriate Metro 2 code representing a positive closure."
    },
    {
        "issue": "Credit inquiries on your report that you don't recognize (LABEL_5)",
        "content": "Complaint: I see three hard inquiries from 'ABC Lending' on Jan 15th. I did not apply for credit with them. \nResolution: Unauthorized hard inquiries harm credit scores. Action: Validate permissible purpose with the inquirer. If fraud or error is confirmed, recode inquiries as 'soft' or delete them entirely."
    },
    {
        "issue": "Investigation took more than 30 days (LABEL_6)",
        "content": "Complaint: I filed a dispute 40 days ago regarding a fraudulent charge, and I still haven't received a final decision. \nResolution: The FCRA generally requires disputes to be resolved within 30 days. Failure to do so is a compliance violation. Action: Expedite the investigation immediately. If the information cannot be verified within the statutory window, the disputed item must be deleted from the file."
    },
    {
        "issue": "Debt is not yours (LABEL_7)",
        "content": "Complaint: A collection agency is calling me about a $200 medical bill for a person named 'John Doe'. My name is 'Jane Smith'. \nResolution: This is a violation of the FDCPA (Fair Debt Collection Practices Act). Action: Cease collection attempts immediately. Mark the debt as disputed and request validation of debt (VOD) from the original creditor. If confirmed as not belonging to the consumer, delete the trade line."
    },
    {
        "issue": "Was not notified of investigation status or results (LABEL_8)",
        "content": "Complaint: You closed my dispute last week, but I never received a letter telling me if you fixed the error or not. \nResolution: Consumers must be provided with the results of the reinvestigation (Notice of Results) within 5 business days of completion. Action: Resend the dispute resolution letter and a free copy of the updated credit report immediately."
    },
    {
        "issue": "Personal information incorrect (LABEL_9)",
        "content": "Complaint: My last name is spelled 'Smyth' on my report, but it is actually 'Smith'. Also, my old address is listed as current. \nResolution: Accuracy of header information is critical for identity verification. Action: Accept the consumer's proof of ID (Driver's License/Utility Bill). Update the name and address fields in the Metro 2 file header."
    },
    {
        "issue": "Other (LABEL_10)",
        "content": "Complaint: The ATM took my card and didn't give it back, and I was late to work because of it. \nResolution: This is a general service or hardware issue. Action: Block the captured card to prevent fraud. Issue a new card immediately via expedited shipping. Apologize for the inconvenience."
    }
]

# Try to get API Key from Streamlit secrets or OS environment
def get_api_key():
    if "GROQ_API_KEY" in st.secrets:
        return st.secrets["GROQ_API_KEY"]
    elif "GROQ_API_KEY" in os.environ:
        return os.environ["GROQ_API_KEY"]
    else:
        # Fallback for local testing if file exists
        try:
            with open("GroqAPI_Key.txt", "r") as f:
                return f.read().strip()
        except:
            return None

@st.cache_resource(show_spinner="Loading Banking Knowledge Base...")
def get_banking_retriever():
    """
    Initializes the Embedding model and Vector Store.
    If a persistent DB exists, it uses it.
    Otherwise, it creates an in-memory knowledge base from SAMPLE_KB_DATA.
    """
    try:
        # 1. Setup Embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # 2. Check if DB exists
        if os.path.exists(PERSIST_DIR):
            # Load existing Vector Store
            vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
        else:
            # CREATE IN-MEMORY KNOWLEDGE BASE
            print("⚠️ Persist directory not found. Building in-memory Knowledge Base from sample data.")
            
            # Convert sample dictionaries to LangChain Documents
            docs = [Document(page_content=item["content"], metadata={"issue": item["issue"]}) for item in SAMPLE_KB_DATA]
            
            # Create ephemeral Chroma vector store
            vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)
        
        # 3. Return Retriever
        return vectorstore.as_retriever(search_kwargs={"k": 3})
        
    except Exception as e:
        print(f"Error loading retriever: {e}")
        return None

def generate_ai_response(user_complaint, category):
    """
    Generates a response using the cached retriever and Groq LLM.
    Args:
        user_complaint (str): The text input from the user.
        category (str): The predicted category label (e.g., 'Debt is not yours').
    """
    
    api_key = get_api_key()
    if not api_key:
        return "⚠️ Error: Groq API Key is missing. Please set it in secrets or environment variables."

    # 1. Get Retriever
    retriever = get_banking_retriever()
    
    # 2. Setup LLM
    try:
        llm = ChatGroq(
            groq_api_key=api_key,
            model_name="llama-3.3-70b-versatile",
            temperature=0.1 
        )
    except Exception as e:
        return f"Error initializing AI Model: {str(e)}"

    # 3. Define Prompt with Contextual Category
    # NOTE: We keep {context} so LangChain can inject the retrieved docs.
    system_prompt = (
        f"You are an expert Banking Resolution Advisor specializing in **{category}** cases.\n"
        "You are an AI system analyzing historical data to provide guidance; you are NOT a bank employee handling the case directly.\n"
        "Analyze the user's complaint and compare it with the retrieved context.\n\n"
        "Provide a suggested resolution plan including:\n"
        "1. Acknowledge the issue with empathy.\n"
        "2. Explain the specific policy or regulation (e.g., FCRA, Reg E) that likely applies based on the context.\n"
        "3. List actionable next steps the customer should take to resolve this.\n"
        "4. Describe what the bank is expected to do under these regulations.\n\n"
        "Important Constraints:\n"
        "- Do NOT use placeholders like '[insert reference number]'.\n"
        "- Do NOT say 'I will investigate'. Instead, say 'The bank is required to investigate'.\n"
        "- Be concise and helpful.\n\n"
        "Context from database:\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # 4. Handle Case where DB failed (Fall back to pure LLM)
    if not retriever:
        # If no vector store, we just use the LLM without RAG context
        chain = prompt | llm
        try:
            response = chain.invoke({"input": user_complaint, "context": "No historical context available."})
            return response.content
        except Exception as e:
            return f"Error generating response: {e}"

    # 5. RAG Execution
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    try:
        response = rag_chain.invoke({"input": user_complaint})
        return response["answer"]
    except Exception as e:
        return f"An error occurred while generating the response: {str(e)}"