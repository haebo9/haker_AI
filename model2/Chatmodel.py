import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import os
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import Sequence, TypedDict, Annotated
from typing_extensions import add_messages
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
import logging

# OpenAI client initialization
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# read_csv file
df = pd.read_csv('/workspaces/haker_AI/model2/sample_data.csv')
target = pd.read_json("/workspaces/haker_AI/model2/input_med.json")["key"].tolist()
print(target)

class ChatModel:
    class State(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]
        language: str

    def __init__(self, model_name="gpt-4o-mini", model_provider="openai"):
        self.model = init_chat_model(model_name, model_provider=model_provider)
        self.llm = ChatOpenAI(model_name=model_name)

        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", """ 당신은 약학 전문가입니다. 어르신에게 설명하듯 친절한 어투로 설명해줘."""),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{last_human_message}"),
            ]
        )

        self.memory = MemorySaver()
        self.chat_message_history = SQLChatMessageHistory(
            session_id="test_session_id", connection="sqlite:///sqlite.db"
        )
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile(checkpointer=self.memory)
        self.chain = self.prompt_template | self.llm
        self.chain_with_history = RunnableWithMessageHistory(
            self.chain,
            lambda session_id: SQLChatMessageHistory(session_id=session_id, connection="sqlite:///sqlite.db"),
            input_messages_key="question",
            history_messages_key="history",
        )

    async def chat(self, message: str, session_id: str = "default_thread"):
        try:
            self.chat_message_history = SQLChatMessageHistory(session_id=session_id, connection="sqlite:///sqlite.db")
            human_message = HumanMessage(content=message)
            self.chat_message_history.add_message(human_message)
            state = {"messages": [human_message], "language": "ko"}
            result = await self.app.ainvoke(state, config={"configurable": {"thread_id": session_id}})
            ai_message = result["messages"][-1]
            self.chat_message_history.add_message(ai_message)
            # 요약 부분만 추출하여 반환
            return self._extract_summary(ai_message.content)
        except Exception as e:
            logging.error(f"Error in chat: {e}")
            raise e
    
    async def _call_model(self, state: State):
            past_messages = self.chat_message_history.messages
            all_messages = past_messages + state["messages"]
            last_human_message = next((msg.content for msg in reversed(all_messages) if isinstance(msg, HumanMessage)), None)
            if last_human_message is None:
                last_human_message = "무엇을 도와드릴까요?"

            selected_row = df[df['제품명'].isin(target)]
            if not selected_row.empty:
                extracted_info = "\n".join([f"{row['제품명']}: {row['효능효과']} {row['용법용량']} {row['주의사항']} {row['상호작용']} {row['부작용']}" for _, row in selected_row.iterrows()])
                last_human_message = f"{last_human_message}\n\n약물 정보:\n{extracted_info}\n\n출처: 식품의약품안전처 (https://www.data.go.kr/data/15075057/openapi.do)"
            
            prompt = self.prompt_template.invoke({
                "history": all_messages,
                "language": state["language"],
                "question": last_human_message,
                "last_human_message": last_human_message,
            })

            try:
                response = await self.model.ainvoke(prompt)
                return {"messages": all_messages + [response]}
            except Exception as e:
                logging.error(f"Error in _acall_model: {e}")
                raise e

    def _build_workflow(self):
            builder = StateGraph(ChatModel.State)
            builder.add_node("model", self._call_model)
            builder.set_entry_point("model")
            builder.add_edge("model", "model")
            return builder