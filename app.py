from dotenv import load_dotenv
from langchain.sql_database import SQLDatabase
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain.document_loaders import UnstructuredFileLoader,TextLoader
import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain.vectorstores import Qdrant,MongoDBAtlasVectorSearch
from pymongo import MongoClient
from langchain.document_loaders import DirectoryLoader
# from langchain_experimental.sql import SQLDatabaseChain

load_dotenv('.env')

llm = ChatOpenAI()
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
        collection_name="test_collection", 
        embeddings=embeddings,
    )
    return vector_store

def connect_mongo_vector_db():
    client = MongoClient('mongodb+srv://Cluster48666:thang4567@cluster48666.ieyo0wo.mongodb.net')
    db = client.langchain_db
    st.write(db.list_collection_names)
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


# Trả lời bằng cách đọc dữ liệu trong DB
def handle_userinput_custom(user_question):

    conn_str = f"mssql+pyodbc://sa:thang4567thang@CPU45\SQL2019:1433/TEST_AI?driver=ODBC+Driver+17+for+SQL+Server"

    db = SQLDatabase.from_uri(conn_str)

    QUERY = """
                Given an input question, first create a syntactically correct sql servers query to run, then look at the results of the query and return the answer.
                When query compare name add N''
                Use the following format:

                Question: Question here
                SQLQuery: SQL Query to run
                Answer: Final answer here

                {question}
            """

    db_chain = SQLDatabaseChain(llm=llm, database=db, verbose=True)

    question = QUERY.format(question=user_question)

    st.write(user_template.replace(
                "{{MSG}}", user_question), unsafe_allow_html=True)

    st.write(bot_template.replace(
                "{{MSG}}", db_chain.run(question)), unsafe_allow_html=True)

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
        handle_userinput(user_question)
        # handle_userinput_custom(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        # raw_documents = TextLoader('test.txt').load()

        # text_splitter = CharacterTextSplitter(separator="\n",
        #                                     chunk_size=1000,
        #                                     chunk_overlap=200,
        #                                     length_function=len)
        # documents = text_splitter.split_documents(raw_documents)
        vector_store = connect_mongo_vector_db()
        query = "Team LMS"
        # docs = vector_store.similarity_search(query)
        
        # vectorstore = FAISS.from_documents(documents,HuggingFaceHubEmbeddings())

        # vector_store.add_documents(documents)
        st.session_state.conversation = get_conversation_chain(vector_store)

if __name__ == '__main__':
    main()
