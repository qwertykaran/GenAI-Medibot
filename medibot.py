import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def main():
    st.set_page_config(page_title="MediBot Chat", layout="wide")
    st.title("🩺 MediBot - Your Medical Assistant")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])
        if message.get("sources"):
            st.markdown("**Sources:**")
            for doc in message["sources"]:
                st.markdown(f"- **{doc.metadata}**: {doc.page_content[:200]}...")

    # Input prompt
    user_input = st.chat_input("Ask your medical question...")

    if user_input:
        st.chat_message('user').markdown(user_input)
        st.session_state.messages.append({'role':'user', 'content': user_input})

        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load vector store")
                return

            GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
            GROQ_MODEL_NAME = "llama-3.1-8b-instant"
            llm = ChatGroq(
                model=GROQ_MODEL_NAME,
                temperature=0.5,
                max_tokens=512,
                api_key=GROQ_API_KEY,
            )

            retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
            combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
            rag_chain = create_retrieval_chain(
                vectorstore.as_retriever(search_kwargs={'k': 3}),
                combine_docs_chain
            )

            with st.spinner("MediBot is thinking... 🤔"):
                response = rag_chain.invoke({'input': user_input})

            answer = response["answer"]
            sources = response.get("context", [])

            st.chat_message('assistant').markdown(answer)
            st.session_state.messages.append({'role':'assistant', 'content': answer, 'sources': sources})

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
