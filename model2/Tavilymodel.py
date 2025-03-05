# -*- coding: utf-8 -*-
"""TAVILY

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1gEU7MdH6zKrG_BheaC1o8yUAh4w_vHwm
"""

# !pip install tavily-python
# !pip install langchain_openai


from tavily import TavilyClient
import streamlit as st
from langchain_openai import ChatOpenAI

tavily_client = TavilyClient(TAVILY_API_KEY)
gpt_4o = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o-mini", temperature=0.1)

st.title("고령자 맞춤 복약 정보 챗봇")

# 사용자 입력 받기
medication_name = "부루펜"

# ✅ Step 1: Tavily API를 활용한 복약 정보 검색 함수 정의
@st.cache_data
def get_medication_info(medication_name):
    try:
        search_response = tavily_client.search(
            query=f"{medication_name} 복약 정보",
            category="health",  # 건강 및 복약 관련 정보 검색
            time_range="month"  # 최근 1개월 동안의 데이터 가져오기
        )

        results = search_response.get("results", [])

        # 위키백과 검색 결과 추가
        wikipedia_search_response = tavily_client.search(
            query=f"{medication_name} 위키백과",
            category="encyclopedia",  # 백과사전 검색
            time_range="month"
        )
        wikipedia_results = wikipedia_search_response.get("results", [])

        results.extend(wikipedia_results)

        if not results:
            return f"'{medication_name}'에 대한 최신 복약 정보를 찾을 수 없습니다. "

        # 검색 결과에서 URL, 제목, 내용을 가져옴
        response_text = f" **'{medication_name}' 복약 정보**\n\n"
        for result in results[:5]:  # 상위 5개 결과만 출력
            response_text += f" **{result['title']}**\n"
            response_text += f" [출처]({result['url']})\n"
            response_text += f" {result.get('content', '요약 정보 없음.')}\n\n"

        return response_text

    except Exception as e:
        return f"❌ 오류 발생: {e}"

# ✅ Step 2: GPT-4o를 활용한 고령자 맞춤형 답변 생성
def generate_chatbot_response(medication_name):
    info = get_medication_info(medication_name)  # ✅ 복약 정보 가져오기

    prompt = f"""
    안녕하세요! 어르신, 복약 정보를 쉽고 친절하게 알려드리는 AI 챗봇입니다.
    '{medication_name}'에 대해 궁금하신가요?

     **아래 내용을 참고해서 설명해 드릴게요.**
    {info}

    ✔ **이 약은 어떤 효과가 있나요?**
    ✔ **어떻게 복용해야 하나요?**
    ✔ **어떤 음식을 피해야 하나요?**
    ✔ **다른 약과 함께 먹어도 괜찮을까요?**

    어르신이 이해하기 쉽도록 친절하고 부드러운 말투로 설명해 드릴게요.
     ✔ **이 약은 어떤 효과/효능 있나요?**
    ✔ **어떻게 복용하면 좋을까요?**
    ✔ **주의할 점은 무엇일까요?**
    ✔ **상호작용 대해 알려주세요**

    **차근차근 설명해 드릴 테니 편하게 읽어주세요! 🧑‍⚕️
    **어르신이 쉽게 이해할 수 있도록 친절하고 부드러운 말투로 안내해 주세요.**
    **문장 끝을 "~해요"로 마무리하고, 이모티콘도 사용해서 편안한 분위기를 만들어 주세요.**
    **사용자가 신뢰할 수 있도록 명확한 정보만 제공해 주세요.**
    **없는 정보는 없습니다.다시 질문해주세요!라고 답변해주세요.**
    **약 정보만 받아주세요**
    **구체적인 상호작용 
    """

    response = gpt_4o.invoke(prompt)
    return {response.content}

if medication_name:
    response = get_medication_info(medication_name)
    st.markdown(response)
    chatbot_response = generate_chatbot_response(medication_name)
    st.markdown(chatbot_response)

    