from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import Sequence
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
import logging

# Load environment variables
load_dotenv()

# Translation and Chat 
class ChatModel:
    class State(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]
        language: str

    def __init__(self, model_name="gpt-4o-mini", model_provider="openai"):
        # Initialize the model for LangGraph and LangChain
        self.model = init_chat_model(model_name, model_provider=model_provider)
        self.llm = ChatOpenAI(model_name=model_name)

        # Set up the prompt for LangChain
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", """
                 You are a helpful assistant. 
                 Answer all questions to the best of your ability in {language}. 
                 When answering questions about professional concepts, 
                 always use English terminology.
                **Important:** You have access to the complete conversation history. 
                 Use this history to provide contextually relevant and coherent answers.
                Remember and maintain the flow of the conversation.
                 if sume one ask you, can you remember the history of talk, 
                 you should answer, YES! """),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),
            ]
        )

        # Use MemorySaver for LangGraph checkpointing
        self.memory = MemorySaver()

        # Set up SQLite-based chat message history
        self.chat_message_history = SQLChatMessageHistory(
            session_id="test_session_id", connection="sqlite:///sqlite.db"
        )

        # Configure LangGraph workflow
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile(checkpointer=self.memory)

        # Configure LangChain's RunnableWithMessageHistory for RAG
        self.chain = self.prompt_template | self.llm
        self.chain_with_history = RunnableWithMessageHistory(
            self.chain,
            lambda session_id: SQLChatMessageHistory(session_id=session_id, connection="sqlite:///sqlite.db"),
            input_messages_key="question",
            history_messages_key="history",
        )

    def _build_workflow(self):
        # Build the LangGraph workflow
        workflow = StateGraph(state_schema=self.State)
        workflow.add_node("model", self._call_model)
        workflow.add_edge(START, "model")
        return workflow
    
    async def chat(self, message: str, session_id: str = "default_thread"):
        """Handles chat messages and returns the model's response."""
        try:
            self.chat_message_history = SQLChatMessageHistory(session_id=session_id, connection="sqlite:///sqlite.db")
            human_message = HumanMessage(content=message)
            self.chat_message_history.add_message(human_message)
            state = {"messages": [human_message], "language": "ko"} # 채팅은 항상 한글로 처리
            result = await self.app.ainvoke(state, config={"configurable": {"thread_id": session_id}})
            ai_message = result["messages"][-1]
            self.chat_message_history.add_message(ai_message)
            return ai_message.content
        except Exception as e:
            logging.error(f"Error in chat: {e}")
            raise e
    
    async def _call_model(self, state: State):
        """Asynchronously calls the model and processes the response. 
        Generates a prompt based on the given state and conversation history, 
        then calls the language model to receive a response."""

        past_messages = self.chat_message_history.messages
        all_messages = past_messages + state["messages"]
        last_human_message = next((msg.content for msg in reversed(all_messages) if isinstance(msg, HumanMessage)), None)
        if last_human_message is None:
            last_human_message = "Hello, how can I help you?"

        prompt = self.prompt_template.invoke({
            "history": all_messages,
            "language": state["language"],
            "question": last_human_message,
        })

        try:
             # Asynchronously invoke the model
            response = await self.model.ainvoke(prompt)
            return {"messages": all_messages + [response]}
        except Exception as e:
            logging.error(f"Error in _acall_model: {e}")
            raise e