import streamlit as st
from dotenv import load_dotenv
from htmlTemplates import css, bot_template, user_template
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceHubEmbeddings
import os, io
# Select embeddings
embeddings = OpenAIEmbeddings()

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_file_text(file_txt):
    text = ""
    for txt in file_txt:
        documents = [txt.read().decode()]
    for doc in documents:
        text += doc
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    return chunks

def get_vectorstore(text_chunks):
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    #llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def get_ask_file_pdf(file_read):
    raw_text = get_pdf_text(file_read)
    text_chunks = get_text_chunks(raw_text)
    vectorstore = get_vectorstore(text_chunks)
    
    # st.session_state.conversation = get_conversation_chain(vectorstore)

def get_ask_file_txt(file_read):
    raw_text = get_file_text(file_read)
    text_chunks = get_text_chunks(raw_text)
    vectorstore = get_vectorstore(text_chunks)
    
    #st.session_state.conversation = get_conversation_chain(vectorstore)

def get_ask_document(file_upload, upload_type_file):
    vectorstore = ""
    if upload_type_file == "File PDF":
        vectorstore = get_ask_file_pdf(file_upload)
    elif upload_type_file == "File Text":
        vectorstore = get_ask_file_txt(file_upload)
    elif upload_type_file == "File Word":
        st.write("read file .doc")
    # st.write(vectorstore)
    st.session_state.conversation = get_conversation_chain(vectorstore)

    # for file in file_upload:
    #     root_ext  = os.path.splitext(file.name)
    #     if root_ext[1] == ".pdf":
    #         st.write("read file .pdf") 
        #     get_ask_file_pdf(file)
        # if root_ext[1] == ".txt":
        #     st.write("read file .txt")
        # if root_ext[1] == ".doc" or root_ext[1] == ".docx":
        #     st.write("read file .doc")
            
    
    

