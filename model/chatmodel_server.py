from fastapi import FastAPI, Query, Response
import logging, json, os
from dotenv import load_dotenv
from chatmodel import ChatModel

# 앱/모델 인스턴스 설정 
app = FastAPI() 
chat_model = ChatModel()

# api_key
load_dotenv()
api_key = os.getenv("API_KEY")

@app.get("/")
def read_root():
    return {"model":"chat"}

@app.get("/chat")
async def chat(message: str = Query(..., description="user message"),
               session_id: str = Query("default_session", description="session ID")):
    try:
        response = await chat_model.chat(message, session_id)
        return {"response": response}
    except Exception as e:
        logging.error(f"Error in /chat: {e}")
        return Response(
            content=json.dumps({"error": str(e)}, ensure_ascii=False, indent=4),
            media_type="application/json",
        )

# $ fastapi dev chatmodel_server.py
