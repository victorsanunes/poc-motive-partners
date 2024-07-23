import sys
sys.path.append('./app/src')
from config import config
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from db_agent import SQLDatabaseAgent

llm = ChatOpenAI(model="gpt-4o", temperature=0)

examples = [
    {"input": 'Get the final earning for clients whose Analyst Rating was to sell the shares',
    "query": '''
    SELECT
        fac.Client_Id,
        fac.Client,
        SUM(ROUND(fac.Quantity * (fac."Current Price"-fac."Buy Price"))) AS FinalEarning
    FROM financial_advisor_clients AS fac
    WHERE
        fac."Analyst Rating" = 'Sell'
    GROUP BY
        fac.Client_Id,
        fac.Client
    ORDER BY
        fac.Client_Id ASC
    '''}
]

def create_sql_database_agent(messages_history, db, toolkit):
    sql_database_agent = SQLDatabaseAgent(
        db=db,
        llm=llm,
        examples=examples,
        message_history=messages_history,
        toolkit=toolkit
    )

    return sql_database_agent
