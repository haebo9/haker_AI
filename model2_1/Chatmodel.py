import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import os
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import Sequence, TypedDict, Annotated
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
import logging
import json

# Configuration
load_dotenv()
ROOT_PATH = "/Users/jaeseoksee/Documents/code/haker_AI/model2/"
DB_CONNECTION = "sqlite:///sqlite.db"
SESSION_ID = "test_session_id"
MODEL_NAME = "gpt-4o"
MODEL_PROVIDER = "openai"
TRIM_LENGTH = 150

# gemini Model import


# Data Loading and Preparation
def load_and_prepare_data():
    df = pd.read_csv(f'{ROOT_PATH}/sample_data.csv')
    target = pd.read_json(f"{ROOT_PATH}/input_med.json")["key"].tolist()
    target_df = df[df['제품명'].isin(target)]
    df_indexed = target_df.set_index('제품명')
    df_indexed.to_json(f"{ROOT_PATH}/target_data.json", orient='index', force_ascii=False, indent=4)

    with open(f"{ROOT_PATH}/target_data.json", 'r', encoding='utf-8') as f:
        target_data_dict = json.load(f)
    with open(f"{ROOT_PATH}/new_data.json", "r", encoding="utf-8") as f:
        new_data = json.load(f)

    return df, target, target_data_dict, new_data

df, target, target_data_dict, new_data = load_and_prepare_data()

# Chat Model Class Definition
class ChatModel:
    class State(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]
        language: str

    def __init__(self, model_name=MODEL_NAME, model_provider=MODEL_PROVIDER):
        self.model = init_chat_model(model_name, model_provider=model_provider)
        self.llm = ChatOpenAI(model_name=model_name)
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", 
                """당신은 약학 전문가입니다. 제공된 약물 정보를 바탕으로 질문에 답변하세요. 
                기존약에 해당하는 데이터가 NaN이면 추론 혹은 사전 데이터를 사용하여 답을 해주세요. 
                새로운 약이 입력되면 새로운 약과 기존 약을 같이 먹어도 문제가 없는지 판단하고, 근거를 설명하세요.
                기존 약 성분:{extracted_info}
                새로운 약 성분: {new_component}

                모든 답은 수치적인 부분을 제거하고 효능과 주의사항을 중심으로 3줄 요약하여 핵심적인 내용만을 담아 응답하라.
                 """
                 ),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{last_human_message}"),
            ]
        )

        self.memory = MemorySaver()
        self.chat_message_history = SQLChatMessageHistory(
            session_id=SESSION_ID, connection=DB_CONNECTION
        )
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile(checkpointer=self.memory)
        self.chain = self.prompt_template | self.llm
        self.chain_with_history = RunnableWithMessageHistory(
            self.chain,
            lambda session_id: [],
            # lambda session_id: SQLChatMessageHistory(session_id=session_id, connection=DB_CONNECTION),
            input_messages_key="question",
            history_messages_key="history",
        )

    async def chat(self, message: str, session_id: str = "default_session"):
        try:
            self.chat_message_history = SQLChatMessageHistory(session_id=session_id, connection=DB_CONNECTION)
            human_message = HumanMessage(content=message)
            self.chat_message_history.add_message(human_message)
            state = {"messages": [human_message], "language": "ko"}
            result = await self.app.ainvoke(state, config={"configurable": {"thread_id": session_id}})
            ai_message = result["messages"][-1]
            self.chat_message_history.add_message(ai_message)
            trimmed_content = self._trim_content(ai_message.content)
            return {trimmed_content.content}
        except Exception as e:
            logging.error(f"Error in chat: {e}")
            raise e
    
    async def _call_model(self, state: State):
            past_messages = self.chat_message_history.messages
            all_messages = past_messages + state["messages"]
            last_human_message = next((msg.content for msg in reversed(all_messages) if isinstance(msg, HumanMessage)), None)
            if last_human_message is None:
                last_human_message = "무엇을 도와드릴까요?"

            extracted_info = self._extract_info()
            new_component = ", ".join(new_data.get("성분", []))
            
            prompt = self.prompt_template.invoke({
                "history": all_messages,
                "language": state["language"],
                "last_human_message": last_human_message,
                "extracted_info": extracted_info,
                "new_component": new_component,
            })

            try:
                response = await self.model.ainvoke(prompt)
                return {"messages": all_messages + [response]}
            except Exception as e:
                logging.error(f"Error in _acall_model: {e}")
                raise e

    def _extract_info(self):
        extracted_info = ""
        selected_row = df[df['제품명'].isin(target)]
        if not selected_row.empty:
            extracted_info += selected_row.to_string() + "\n\n"

        for drug in target:
            if drug in target_data_dict:
                drug_details = target_data_dict[drug]
                info_lines = [f"{key}: {value if value is not None else '정보 없음'}" for key, value in drug_details.items()]
                extracted_info += "\n".join(info_lines) + "\n\n"
            else:
                extracted_info += f"{drug}: 정보 없음\n\n"
        return extracted_info

    def _build_workflow(self):
            builder = StateGraph(ChatModel.State)
            builder.add_node("model", self._call_model)
            builder.set_entry_point("model")
            return builder

    def _trim_content(self, content: str) -> str:
        """Trims the content to a specified length."""
        if len(content) > TRIM_LENGTH:
            return content[:TRIM_LENGTH] + "..."
        return content

