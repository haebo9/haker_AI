from fastapi import FastAPI, Query, Response, Header
import logging, json, os
from dotenv import load_dotenv
from Chatmodel import ChatModel
from openai import OpenAI

# 앱/모델 인스턴스 설정 
app = FastAPI() 
chat_model = ChatModel()

# api_key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY") # api key를 전역변수로 설정
client = OpenAI(api_key=api_key)

@app.get("/")
def read_root():
    return {"model":"chat"}

@app.get("/chat")
async def chat(message: str = Query(..., description="user message"),
               session_id: str = Query("default_session", description="session ID"),
               api_key_header: str = api_key): # 헤더에서 api키를 받는다.
    if api_key_header != api_key: # api키가 일치하지 않으면 401에러를 낸다.
        return Response(status_code=401, content=json.dumps({"error": "Unauthorized"}), media_type="application/json")
    try:
        response = await chat_model.chat(message, session_id)
        return {"response": response}
    except Exception as e:
        logging.error(f"Error in /chat: {e}")
        return Response(
            content=json.dumps({"error": str(e)}, ensure_ascii=False, indent=4),
            media_type="application/json",
        )

# $ fastapi dev Chatmodel_server.py