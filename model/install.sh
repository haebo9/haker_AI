#!/bin/bash
# github Codespaces(서버) 환경에서 실행

# 시스템 업데이트 및 pip 업그레이드
echo "$SYSTEM_INSTALL"
python3 -m pip install --upgrade pip

# LangChain 및 OpenAI 관련 라이브러리 설치
echo "$LANGCHAIN_OPENAI_INSTALL"
pip install langchain
pip install -qU "langchain[openai]"

# 환경 변수 설정 (dotenv)
echo "$ENV_INSTALL"
pip install dotenv

# FastAPI (서버 통신)
echo "$SERVER_INSTALL"
pip install "fastapi[standard]"

# Tavily AI 및 추가 LangChain 라이브러리 설치
echo "$TAVILY_INSTALL"
pip install -U langchain-community langgraph langchain-anthropic tavily-python langgraph-checkpoint-sqlite

# 실행코드: $ bash install.sh
