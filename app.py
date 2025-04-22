import os
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA



def load_and_index_pdf(pdf_path: str) -> FAISS:
    """
    Loads the MITRE ATT&CK PDF, splits its content into manageable chunks,
    creates OpenAI embeddings for each chunk, and builds a FAISS vector store.
    
    Adjust the chunk_size and chunk_overlap for higher retrieval accuracy.
    """
   
    loader = PyPDFLoader(pdf_path)
    raw_documents = loader.load()
    
   
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    documents = text_splitter.split_documents(raw_documents)
    
   
    embeddings = OpenAIEmbeddings()
    
    
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store

def create_retrieval_chain(vector_store: FAISS) -> RetrievalQA:
    """
    Creates a RetrievalQA chain with a retriever using the FAISS vector store.
    Retrieves the top k passages (tuned for quality) and uses them as context to ask the LLM.
    """
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o")
    
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain

def analyze_threat(chain: RetrievalQA, title: str, description: str) -> dict:
    """
    Given a threat alert title and/or description, form a query that instructs the model
    to identify the most relevant MITRE ATT&CK tactic and technique with reasoning.
    """
    
    query = (
        f"Threat alert details:\nTitle: {title}\nDescription: {description}\n\n"
        "Based on the above and using the local MITRE ATT&CK database, identify the most relevant "
        "MITRE tactic with their respective IDs and technique with their respective IDs. Provide detailed reasoning behind your choice, explaining how the "
        "alert correlates with the tactics and techniques. Put the generated content in tabular form and there should be no other text after the table"
    )
    result = chain(query)
    return result



def main():
    st.title("MITRE ATT&CK Classification with AI")
    
   
    st.sidebar.header("Configuration")
    openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    pdf_path = st.sidebar.text_input("MITRE ATT&CK PDF Path", value="mitre.pdf")
    
   
    if not openai_api_key:
        st.warning("Please enter your OpenAI API Key in the sidebar to continue.")
        return
    else:
        os.environ["OPENAI_API_KEY"] = openai_api_key
    
    
    @st.cache_resource(show_spinner=False)
    def get_vector_store():
        return load_and_index_pdf(pdf_path)
    
    
    if st.sidebar.button("Load and Index MITRE Database"):
        with st.spinner("Processing PDF and building vector store..."):
            vector_store = get_vector_store()
            qa_chain = create_retrieval_chain(vector_store)
            st.session_state.qa_chain = qa_chain
        st.success("MITRE ATT&CK database loaded and indexed successfully!")
    
    
    if "qa_chain" in st.session_state:
        st.header("Threat Alert Analysis")
        title = st.text_input("Enter Threat Alert Title ")
        description = st.text_area("Enter Threat Alert Description ")
    
        if st.button("Analyze Threat"):
            if not title and not description:
                st.error("Please provide at least a title or description for the threat alert.")
            else:
                with st.spinner("Analyzing threat alert..."):
                    result = analyze_threat(st.session_state.qa_chain, title, description)
                st.subheader("Analysis Result")
                st.write(result["result"])
                
                
                with st.expander("Show source documents used for analysis"):
                    for i, doc in enumerate(result.get("source_documents", []), start=1):
                        st.markdown(f"**Document {i} (Page info: {doc.metadata.get('page', 'N/A')})**")
                        st.write(doc.page_content)
    else:
        st.info("Load the MITRE ATT&CK database from the sidebar to get started.")

if __name__ == "__main__":
    main()
