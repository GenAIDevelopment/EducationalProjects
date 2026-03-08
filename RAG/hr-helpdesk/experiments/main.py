import os
import streamlit as st
from retriever import HRRetrievalPipeline, RetrievalConfig
from langchain_google_vertexai import VertexAI, VertexAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()



def get_llm():
    return VertexAI(model_name=os.getenv('MODEL', 'gemini-2.5-flash-lite'))


st.set_page_config(
    page_icon="🔬",
    page_title="HR Helpdesk RAG", 
    initial_sidebar_state="expanded",
    layout="wide")

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """ You are an HR Helpdesk assistant
expert. Answer the question based on the context provided.
     If you don't know the answer, just say you don't know.
     Don't try to make up an answer.
     Use the context provided to answer the question.
     If the context is empty, just say 'No context provided'.
     """),
    ("human", "Context:\n{context}\n\nQuestion: {question}")
])

def get_pipeline() -> HRRetrievalPipeline:
    config = RetrievalConfig(
        connection=os.getenv("DB_CONNECTION_STRING", 'postgresql://admin:admin123@localhost:5432/vectordb'),
        collection_name="hr_helpdesk",
        embedding_model=os.getenv('EMBEDDING_MODEL', 'text-embedding-005')
    )
    return HRRetrievalPipeline(config)

def get_answer(question: str) -> str:
    pipeline = get_pipeline()
    retrieval_result = pipeline.retrieve(question)
    context = "\n\n".join([doc.page_content for doc in retrieval_result.docs])
    prompt = ANSWER_PROMPT.format(context=context, question=question)
    llm = get_llm()
    return llm.invoke(prompt)

def main():
    st.title("🔬 HR Helpdesk RAG")
    st.caption("Ask me anything about HR policies and procedures!")

    pipeline = get_pipeline()
    with st.sidebar:
        st.subheader("Retrieval Config")
        st.write(pipeline.config)
    user_query = st.chat_input("What's your question?", key="query")
    if user_query:
        with st.spinner("Thinking..."):
            answer = get_answer(user_query)
        st.write(answer)


if __name__ == "__main__":
    main()
