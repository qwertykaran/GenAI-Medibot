import os
from dotenv import load_dotenv

# Updated imports
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Load environment variables
load_dotenv()

# Step 1: Setup Groq LLM
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables. Add it to your .env file.")

GROQ_MODEL_NAME = "llama-3.1-8b-instant"  # Change if needed

print("Initializing Groq LLM...")
llm = ChatGroq(
    model=GROQ_MODEL_NAME,
    temperature=0.5,
    max_tokens=512,
    api_key=GROQ_API_KEY,
)

# Step 2: Connect LLM with FAISS
DB_FAISS_PATH = "vectorstore/db_faiss"
print(f"Loading FAISS vectorstore from {DB_FAISS_PATH}...")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

try:
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    print("FAISS vectorstore loaded successfully!")
except Exception as e:
    raise RuntimeError(f"Error loading FAISS vectorstore: {e}")

# Step 3: Build RAG chain
print("Pulling retrieval-qa-chat prompt from LangChain Hub...")
retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

print("Creating chains...")
combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
rag_chain = create_retrieval_chain(db.as_retriever(search_kwargs={'k': 3}), combine_docs_chain)

print("Ready to accept queries!\n")

# Step 4: Query loop
while True:
    user_query = input("Write Query Here (type 'exit' or 'quit' to stop): ").strip()
    if user_query.lower() in ['exit', 'quit']:
        print("Exiting...")
        break

    print("\nProcessing query, please wait...")
    try:
        response = rag_chain.invoke({'input': user_query})
        print("\nRESULT: ", response["answer"])
        print("\nSOURCE DOCUMENTS:")
        for doc in response.get("context", []):
            print(f"- {doc.metadata} -> {doc.page_content[:200]}...")
        print("\n" + "-"*50 + "\n")
    except Exception as e:
        print(f"Error processing query: {e}")
