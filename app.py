from dotenv import load_dotenv
from langchain.sql_database import SQLDatabase
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain,create_sql_query_chain
from htmlTemplates import css, bot_template, user_template
from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain.document_loaders import UnstructuredFileLoader,TextLoader
import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain.vectorstores import Qdrant,MongoDBAtlasVectorSearch
from pymongo import MongoClient
from langchain.document_loaders import DirectoryLoader
from langchain_experimental.sql import SQLDatabaseChain
from PyPDF2 import PdfReader
from langchain import hub
from langchain.prompts import PromptTemplate

load_dotenv('.env')

llm = ChatOpenAI(model="gpt-3.5-turbo-16k",temperature=0)

embeddings = HuggingFaceHubEmbeddings()
# Kết nối vector DB
def connect_qdrant_vector_db():
    client = QdrantClient("localhost", port=6333)
    # client.recreate_collection(
    #     collection_name="test_collection",
    #     vectors_config=VectorParams(size=768, distance=Distance.DOT),
    # )
    vector_store = Qdrant(
        client=client, 
        collection_name="test_collectionz", 
        embeddings=embeddings,
    )
    return vector_store

def connect_mongo_vector_db():
    client = MongoClient('mongodb+srv://Cluster48666:thang4567@cluster48666.ieyo0wo.mongodb.net')
    db_name = "langchain_db"
    collection_name = "langchain_col"
    collection = client[db_name][collection_name]
    index_name = "langchain_demo"
    vector_store = MongoDBAtlasVectorSearch(
        collection, embeddings, index_name=index_name
    )
    return vector_store

def get_conversation_chain(vectorstore):
    
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

# Trả lời bình thường
def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})

    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Trả lời bằng cách đọc dữ liệu trong DB
def handle_userinput_custom(user_question):

    conn_str = f"mssql+pyodbc://vnr:rUbTwiQ8Rb6OEL4@115.73.215.48,16968:1433/LMS_MISA_TEST?driver=ODBC+Driver+17+for+SQL+Server"

    db = SQLDatabase.from_uri(conn_str,
                              sample_rows_in_table_info=1,
                              include_tables=["mdl_user","mdl_course","mdl_course_completions","mdl_course_modules",
                                              "mdl_course_modules_completion","mdl_modules","mdl_quiz","mdl_quiz_attempts",
                                              "mdl_course_categories","mdl_user_enrolments","mdl_enrol","mdl_role_assignments"])

    QUERY = """
                Given an input question, first create a syntactically correct sql servers query to run, then look at the results of the query and return the answer.
                When query compare name add N''
                Use the following format:

                Question: Question here
                SQLQuery: SQL Query to run
                Answer: Final answer here

                {question}
            """ 
    db_chain = SQLDatabaseChain(llm=llm, database=db, verbose=False)
    response = QUERY.format(question=user_question)

    # chain = create_sql_query_chain(ChatOpenAI(model="gpt-3.5-turbo-16k",temperature=0,max_tokens='4000'), db)
    # response = chain.invoke({"question":user_question})
    st.write(user_template.replace(
                "{{MSG}}", user_question), unsafe_allow_html=True)
    st.write(bot_template.replace(
                "{{MSG}}", db_chain.run(response)), unsafe_allow_html=True)
    # st.write("Sqlquery Exe: " + response)

def main():

    st.set_page_config(page_title="ChatBot",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    user_question = st.text_input("Ask a question:")
    if user_question:
        # handle_userinput(user_question)
        handle_userinput_custom(user_question)
    with st.sidebar:
        st.subheader("Your documents")
        # pdf_docs = st.file_uploader(
        #     "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        # if st.button("Process"):
        #     with st.spinner("Processing"):
                # st.write(pdf_docs)
                # raw_documents = TextLoader('test.txt').load()

                # text_splitter = CharacterTextSplitter(separator="\n",
                #                                     chunk_size=1000,
                #                                     chunk_overlap=200,
                #                                     length_function=len)
                # documents = text_splitter.split_documents(raw_documents)
                # vector_store = connect_mongo_vector_db()

                # st.session_state.conversation = get_conversation_chain(vector_store)

if __name__ == '__main__':
    main()
