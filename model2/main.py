import uvicorn
import Chatmodel_server

app = Chatmodel_server.app

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

# uvicorn main:app --reload