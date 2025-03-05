from fastapi import FastAPI, Query, Response
import logging
import json
from model.ChatModel import ChatModel

# 앱/모델 인스턴스 설정
app = FastAPI()

# ChatModel 인스턴스 생성 (인수 없음)
chatmodel = ChatModel()


@app.get("/")
def read_root():
    return {"model": "chat"}


@app.get("/chat")
async def chat(message: str = Query(..., description="user message"),
               session_id: str = Query("default_session", description="session ID")):
    try:
        response = await chatmodel.chat(message, session_id)
        return {"response": response}
    except Exception as e:
        logging.error(f"Error in /chat: {e}")
        return Response(
            content=json.dumps({"error": str(e)}, ensure_ascii=False, indent=4),
            media_type="application/json",
        )