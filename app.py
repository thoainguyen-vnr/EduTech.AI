import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain import OpenAI, SQLDatabase, SQLDatabaseChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain.document_loaders import UnstructuredFileLoader,TextLoader

load_dotenv('.env')
llm = ChatOpenAI()

def get_conversation_chain(vectorstore):
    
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})

    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", user_question), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", '123567'), unsafe_allow_html=True)


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
        handle_userinput_custom(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        # raw_documents = TextLoader('test.txt').load()
        # raw_documents = UnstructuredFileLoader('document/Cựu bộ trưởng Y tế Nguyễn Thanh Long bị cáo buộc nhận hơn 2 triệu USD trong vụ Việt Á.html').load()
        # text_splitter = CharacterTextSplitter(separator="\n",
        #                                     chunk_size=1000,
        #                                     chunk_overlap=200,
        #                                     length_function=len)
        # documents = text_splitter.split_documents(raw_documents)
        # vectorstore = FAISS.from_documents(documents,HuggingFaceHubEmbeddings())

        # st.session_state.conversation = get_conversation_chain(vectorstore)

        st.session_state.conversation = []


if __name__ == '__main__':
    main()
