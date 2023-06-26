import streamlit as st 
from langchain.chains import LLMChain, SimpleSequentialChain 
from langchain.llms import OpenAI 
from langchain.prompts import PromptTemplate 
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA


st.title("✅ LangChain Hopsworks Feature Store")


option = st.selectbox(
    'Select',
    ('A', 'B', 'C'))

st.write(option)



persist_directory = "./"


uploaded_file = st.file_uploader("Upload a file", "pdf")
if uploaded_file is not None:
    filebytes = uploaded_file.getvalue()
    filename = "doc.pdf"
    with open(filename, 'wb') as f: 
        f.write(filebytes)

    loader = PyMuPDFLoader(filename)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=10)
    texts = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(documents=texts, 
                                    embedding=embeddings,
                                    persist_directory=persist_directory)
    vectordb.persist()

    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(model_name='gpt-3.5-turbo')

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)


    placeholder = "You chose " + option + ". " 
    st.write(placeholder)

    query = f"###Prompt {placeholder}"
    try:
        llm_response = qa(query)
        st.write(llm_response["result"])
    except Exception as err:
        st.write('Exception occurred. Please try again', str(err))


