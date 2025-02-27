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

# OpenAI client initialization
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# read_csv file
df = pd.read_csv('/workspaces/haker_AI/model2/sample_data.csv')
target = pd.read_json("/workspaces/haker_AI/model2/input_med.json")["key"].tolist() # 리스트 형태
target_df = df[df['제품명'].isin(target)]
df_indexed = target_df.set_index('제품명')
df_indexed.to_json("/workspaces/haker_AI/model2/target_data.json", orient='index', force_ascii=False, indent=4)

# load json2dict data
json_file_path = "/workspaces/haker_AI/model2/target_data.json"
with open(json_file_path, 'r', encoding='utf-8') as f:
    target_data_dict = json.load(f)

# chat model class definition
class ChatModel:
    class State(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]
        language: str

    def __init__(self, model_name="gpt-4o", model_provider="openai"):
        self.model = init_chat_model(model_name, model_provider=model_provider)
        self.llm = ChatOpenAI(model_name=model_name)

        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", """당신은 약학 전문가입니다. 제공된 약물 정보를 바탕으로 질문에 답변하세요. 
                 데이터가 NaN 일시 추론 혹은 사전 데이터를 사용하여 답을 하라.
                 제공된 약물 정보는 다음과 같습니다: {extracted_info}"""),
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

    async def chat(self, message: str, session_id: str = "default_session"):
        try:
            self.chat_message_history = SQLChatMessageHistory(session_id=session_id, connection="sqlite:///sqlite.db")
            human_message = HumanMessage(content=message)
            self.chat_message_history.add_message(human_message)
            state = {"messages": [human_message], "language": "ko"}
            result = await self.app.ainvoke(state, config={"configurable": {"thread_id": session_id}})
            ai_message = result["messages"][-1]
            self.chat_message_history.add_message(ai_message)
            return {"content":ai_message}
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
            extracted_info = ""
            if not selected_row.empty:
                extracted_info += selected_row.to_string() + "\n\n"

            for drug in target:
                if drug in target_data_dict:
                    drug_details = target_data_dict[drug]
                    info_lines = [f"{key}: {value if value is not None else '정보 없음'}" for key, value in drug_details.items()]
                    extracted_info += "\n".join(info_lines) + "\n\n"
                else:
                    extracted_info += f"{drug}: 정보 없음\n\n"
            # print(extracted_info)

            prompt = self.prompt_template.invoke({
                "history": all_messages,
                "language": state["language"],
                "question": last_human_message,
                "last_human_message": last_human_message,
                "extracted_info": extracted_info,
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
            return builder