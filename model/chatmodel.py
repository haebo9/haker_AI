import pandas as pd
from openai import OpenAI
import numpy as np
from dotenv import load_dotenv
import os
from sklearn.metrics.pairwise import cosine_similarity
import pickle
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
import asyncio
import openai

# OpenAI client initialization
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load embeddings from pickle file
with open('drug_embeddings.pkl', 'rb') as f:
    df = pickle.load(f)

# print(df.columns)

item_names = df['제품명'].tolist()

import openai

def test_text(prompt):
    """주어진 텍스트에서 약 이름을 정확하고 표준화된 이름으로 수정합니다. 
    약 이름이 오타이거나 불분명한 경우 올바른 이름을 제공합니다. 
    언급된 약이 존재하지 않거나 약이 아닌 경우 약이 아님을 명시합니다."""
    client = openai.OpenAI()
    messages = [
        {"role": "system", "content": "약 이름을 수정하는 전문가입니다."},
        {"role": "user", "content": prompt}
    ]
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0,
    )
    return response.choices[0].message.content.strip()

# Integrated ChatModel class
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
                당신은 약학 전문가입니다. 
                제공된 약물 정보를 기반으로 질문에 답하고, 필요한 경우 추가적인 의학 정보를 제공하세요.
                답변은 항상 한국어로 작성하며, 전문 용어는 영어로 표기합니다.
                **중요:** 이전 대화 내용을 참고하여 답변해야 합니다.
                대화의 흐름을 유지하고, 맥락에 맞는 답변을 제공하세요.
                만약, 대화 내용을 기억할 수 있는지 질문받으면, "네, 기억하고 있습니다!"라고 답변하세요.
                """),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{last_human_message}"),
            ]
        )

        # Use MemorySaver for LangGrapㄴh checkpointing
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
            state = {"messages": [human_message], "language": "ko"} # Process chat in Korean
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
        last_human_message = test_text(last_human_message)
        if last_human_message is None:
            last_human_message = "Hello, how can I help you?"
        
        # Drug information retrieval (part of the first code)
        query_embedding = client.embeddings.create(input=last_human_message, model="text-embedding-ada-002").data[0].embedding
        similarities = cosine_similarity(np.array(query_embedding).reshape(1, -1), np.array(df['제품명_embedding'].tolist()))
        most_similar_index = np.argmax(similarities)
        most_similar_item = item_names[most_similar_index]
        selected_row = df[df['제품명'] == most_similar_item]
        print(selected_row)

        if not selected_row.empty:
            # Extract retrieved drug information
            extracted_info = selected_row.iloc[0]['효능효과'] + " " + selected_row.iloc[0]['용법용량'] + " " + selected_row.iloc[0]['주의사항'] + " " + selected_row.iloc[0]['상호작용'] + " " + selected_row.iloc[0]['부작용']
            last_human_message = f"{last_human_message}\n\nDrug Information:\n{extracted_info}\n\n출처: 식품의약품안전처 (https://www.data.go.kr/data/15075057/openapi.do)"
        
        prompt = self.prompt_template.invoke({
            "history": all_messages,
            "language": state["language"],
            "question": last_human_message,
            "last_human_message": last_human_message, # last_human_message 추가
        })

        try:
             # Asynchronously invoke the model
            response = await self.model.ainvoke(prompt)
            return {"messages": all_messages + [response]}
        except Exception as e:
            logging.error(f"Error in _acall_model: {e}")
            raise e
