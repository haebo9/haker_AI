import google.generativeai as genai
import json
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

# Gemini Model Code
class GeminiModel:
    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash"):
        self.api_key = api_key
        self.model_name = model_name
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)

    def generate_content(self, prompt: str) -> str:
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Error generating content: {e}")
            return None

    def generate_content_stream(self, prompt: str):
        try:
            response = self.model.generate_content(prompt, stream=True)
            for chunk in response:
                yield chunk.text
        except Exception as e:
            print(f"Error generating content stream: {e}")
            yield f"Error: {e}"

def load_json_file(filepath: str):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {filepath}")
        return None

# Langchain and Data Loading Code
load_dotenv()
ROOT_PATH = "/Users/jaeseoksee/Documents/code/haker_AI/model2/"
DB_CONNECTION = "sqlite:///sqlite.db"
SESSION_ID = "test_session_id"
MODEL_NAME = "gpt-4o"
MODEL_PROVIDER = "openai"
TRIM_LENGTH = 150

def load_and_prepare_data(ROOT_PATH):
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

df, target, target_data_dict, new_data = load_and_prepare_data(ROOT_PATH)

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
            lambda session_id: SQLChatMessageHistory(session_id=session_id, connection=DB_CONNECTION),
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
            
            return {ai_message.content}
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
        if len(content) > TRIM_LENGTH:
            return content[:TRIM_LENGTH] + "..."
        return content

if __name__ == "__main__":
    api_key = "AIzaSyApArsRuO98xud_37C0j9WtJuJqNjwm6f0"
    gemini_model = GeminiModel(api_key)

    text1 = load_and_prepare_data(ROOT_PATH)
    text2 = "부루펜은 이부프로펜 성분의 비스테로이드성 항염증제로, 두통, 치통, 생리통 등 다양한 통증 완화 및 해열 효과가 있으며, 성인은 1회 1정씩 필요시 하루 3회까지 복용 가능하나, 위장 장애, 알레르기 반응, 간 기능 이상 등의 부작용과 임산부 및 어린이 복용 시 주의가 필요하고, 다른 약물과의 상호작용 가능성이 있으므로 의사나 약사와 상담 후 복용해야 합니다."
    prompt = f"""
    첫 번째 글:
    {text1}

    두 번째 글:
    {text2}

    찻반쩨 글은 기존에 먹던 약이고, 두번째 글에 있는 새로운 약을 먹을까 한다. 상호작용을 자세하게 분석하여 두 문장 이내로 설명하라.
    반드시 요약하여 핵심적인 문장으로 설명하라.
    """

    response_text = gemini_model.generate_content(prompt)
    if response_text:
        print("Generated Response:")
        print(response_text)

    # print("\nStreaming Response:")
    # for chunk in gemini_model.generate_content_stream(prompt):
    #     if chunk:
    #         print(chunk, end="")
    # print()

    # Langchain 모델 실행 부분
    chat_model = ChatModel()  # ChatModel 인스턴스 생성
    
    langchain_response = chat_model.chat(message)
    print("\nLangchain Response:")
    print(langchain_response)

