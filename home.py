import streamlit as st
from htmlTemplates import css, bot_template, user_template
from fn_ask_sql_database import handle_userinput_sql_database, config_sql_database
from fn_ask_upload_file import get_ask_document
from fn_get_type_file import get_file_upload

def get_data_source():
        rdo_datasource = st.radio(
            "Please choose question data source!",
            ["Upload file", "SQL database"],
            captions = ["Ask a question Upload file", "Ask a question SQL database"],
            horizontal = True,)
        return rdo_datasource

def get_type_file():
        rdo_typefile = st.radio(
            "Please choose type file upload!",
            ["File PDF", "File Text", "File Word"],
            captions = [".pdf", ".txt", ".docx"],
            horizontal = True,)
        return rdo_typefile

def ask_question_user(question_data_source, file_upload, radio_type_file):
    user_question = st.text_input("Ask a question: " + question_data_source, placeholder="Input search")  
    if user_question:
        if question_data_source == "SQL database":
            ask_question_sqldatabase(user_question) 
            handle_userinput(user_question)
        elif question_data_source == "Upload file":
            handle_userinput(user_question)
     
def ask_question_sqldatabase(user_question):
    res = handle_userinput_sql_database(user_question)
    st.write(user_template.replace(
        "{{MSG}}", user_question), unsafe_allow_html=True)
    st.write(bot_template.replace(
        "{{MSG}}", config_sql_database.run(res)), unsafe_allow_html=True)
    return res

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

def get_radio_type_file(upload_type_file):
    if upload_type_file == "File PDF":
        type_upload = "pdf"
    elif upload_type_file == "File Text":
        type_upload = "txt"
    elif upload_type_file == "File Word":
        type_upload = "doc"
    return type_upload
def main():

    st.set_page_config(page_title="ChatBot",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    question_data_source = get_data_source()
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    radio_type_file = "pdf"
    with st.sidebar:
        upload_type_file = get_type_file()
        radio_type_file = get_radio_type_file(upload_type_file)
        st.subheader("Your documents")
        file_uploader = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True, type=radio_type_file)
        if st.button("Process"):
            with st.spinner("Processing"):
                 get_ask_document(file_uploader, radio_type_file)
    rs_ask = ask_question_user(question_data_source, file_uploader, radio_type_file)

if __name__ == '__main__':
    main()