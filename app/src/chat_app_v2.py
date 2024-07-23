import sys

import engines.setup
sys.path.append('./app/src')
from config import config
import engines
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.utilities import SQLDatabase
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
import streamlit as st

st.set_page_config(page_title="Financial Copilot", page_icon="💰")
st.title("💰 Financial Advisor Copilot")

# Set up memory
msgs = StreamlitChatMessageHistory(key="langchain_messages")
if st.session_state.get("messages_history") is None:
    st.session_state.messages_history = ChatMessageHistory()

# Create agent
llm = llm = ChatOpenAI(model="gpt-4o", temperature=0)

db_path = config.ARTIFACTS_PATH / 'sql_database.db'
db = SQLDatabase.from_uri(f'sqlite:///{db_path}')
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
agent = engines.setup.create_sql_database_agent(
    messages_history=st.session_state.messages_history, 
    db=db,
    toolkit=toolkit
)
if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you?")

view_messages = st.expander("View the message contents in session state")
openai_api_key = config.CONFIG['openai']['token']

# Render current messages from StreamlitChatMessageHistory
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

# If user inputs a new prompt, generate and draw a new response
if prompt := st.chat_input():
    st.chat_message("human").write(prompt)
    
    # Note: new messages are saved to history automatically by Langchain during run
    config = {"configurable": {"session_id": "any"}}
    response = agent.get_answer(input_message=prompt)
    st.chat_message("ai").write(response.get("output"))

# Draw the messages at the end, so newly generated ones show up immediately
with view_messages:
    """
    Message History initialized with:
    ```python
    msgs = StreamlitChatMessageHistory(key="langchain_messages")
    ```

    Contents of `st.session_state.langchain_messages`:
    """
    view_messages.json(st.session_state.langchain_messages)