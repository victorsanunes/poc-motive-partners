from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.vectorstores import FAISS
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

class SQLDatabaseAgent():
    def __init__(self, db, llm, examples, message_history=None):
        self.db = db
        self.llm = llm
        self.system_prefix  = """
            You are an agent designed to interact with a SQL database.
            Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
            Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.
            You can order the results by a relevant column to return the most interesting examples in the database.
            Never query for all the columns from a specific table, only ask for the relevant columns given the question.
            You have access to tools for interacting with the database.
            Only use the given tools. Only use the information returned by the tools to construct your final answer.
            You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

            DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

            If the question does not seem related to the database, just return "I don't know" as the answer.

            Here are some examples of user inputs and their corresponding SQL queries:
        """
        self.examples = examples
        self.example_selector = self._set_example_selector(self.examples)
        self.few_shot_prompt = self._set_few_shot_prompt(self.example_selector)
        self.full_prompt = self._set_full_prompt(self.few_shot_prompt)
        if message_history is None:
            self.message_history = ChatMessageHistory()
        else:
            self.message_history = message_history
        self.agent_executor = create_sql_agent(
            llm, 
            db=db, 
            prompt=self.full_prompt, 
            agent_type="openai-tools", 
            verbose=True
        )
        self.agent_with_chat_history = RunnableWithMessageHistory(
            self.agent_executor,
            lambda session_id: self.message_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )    
    
    def _set_example_selector(self, examples):
        example_selector = SemanticSimilarityExampleSelector.from_examples(
            examples,
            OpenAIEmbeddings(),
            FAISS,
            k=5,
            input_keys=["input"],
        )
        
        return example_selector

    def _set_few_shot_prompt(self, example_selector):
        few_shot_prompt = FewShotPromptTemplate(
            example_selector=example_selector,
            example_prompt=PromptTemplate.from_template(
                "User input: {input}\nSQL query: {query}"
            ),
            input_variables=["input", "dialect", "top_k"],
            prefix=self.system_prefix,
            suffix="",
        )

        return few_shot_prompt

    def _set_full_prompt(self, few_shot_prompt):
        full_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate(prompt=few_shot_prompt),
                ("human", "{input}"),
                MessagesPlaceholder("agent_scratchpad"),
            ]
        )

        return full_prompt

    def get_answer(self, input_message):
        # prompt_val = self.full_prompt.invoke(
        #     {
        #         "input": input_message,
        #         "top_k": 5,
        #         "dialect": "SQLite",
        #         "agent_scratchpad": [],
        #     }
        # )

        return self.agent_with_chat_history.invoke(
            {
                "input": input_message
            },
            # This is needed because in most real world scenarios, a session id is needed
            # It isn't really used here because we are using a simple in memory ChatMessageHistory
            config={"configurable": {"session_id": "<foo>"}},
        )

    
