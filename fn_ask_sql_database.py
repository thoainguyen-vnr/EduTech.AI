from dotenv import load_dotenv
from langchain.sql_database import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.chat_models import ChatOpenAI


load_dotenv('.env')
llm = ChatOpenAI(model="gpt-3.5-turbo-16k",temperature=0)
def config_sql_database():
    conn_str = f"mssql+pyodbc://vnr:rUbTwiQ8Rb6OEL4@115.73.215.48,16968:1433/LMS_MISA_TEST?driver=ODBC+Driver+17+for+SQL+Server"
    cof_db = SQLDatabase.from_uri(conn_str,
                              sample_rows_in_table_info=1,
                              include_tables=["mdl_user","mdl_course","mdl_course_completions","mdl_course_modules",
                                              "mdl_course_modules_completion","mdl_modules","mdl_quiz","mdl_quiz_attempts",
                                              "mdl_course_categories","mdl_user_enrolments","mdl_enrol","mdl_role_assignments"])
    return cof_db

def handle_userinput_sql_database(user_question):
    db = config_sql_database()
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
    response = QUERY.format(question=user_question)
    return db_chain.run(response)