import pandas as pd
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dotenv import load_dotenv
import os

# OpenAI 클라이언트 초기화
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 데이터프레임 생성 (예시)
data = {'text': ["OpenAI는 인공지능 연구소입니다.", "임베딩은 텍스트를 벡터로 변환하는 기술입니다.", "RAG는 검색과 생성을 결합한 기술입니다."], 'category': ['AI', 'NLP', 'NLP']}
df = pd.DataFrame(data)

# 텍스트 청크 분할 (예시)
chunks = df['text'].tolist()

# 임베딩 생성
embeddings = []
for chunk in chunks:
    response = client.embeddings.create(input=chunk, model="text-embedding-ada-002")
    embeddings.append(response.data[0].embedding)

# 벡터 데이터베이스 저장 (예시)
vector_db = dict(zip(chunks, embeddings))

# 사용자 질문 임베딩
query = "OpenAI가 춤을 출수 있나??"
query_embedding = client.embeddings.create(input=query, model="text-embedding-ada-002").data[0].embedding

# 유사도 계산 및 검색
similarities = []
for chunk, embedding in vector_db.items():
    similarity = cosine_similarity(np.array(query_embedding).reshape(1, -1), np.array(embedding).reshape(1, -1))[0][0]
    similarities.append((similarity, chunk))

similarities.sort(reverse=True)
retrieved_chunks = [chunk for _, chunk in similarities[:2]]

# 언어 모델(LLM)에 검색된 청크 입력 및 답변 생성 (예시)
prompt = f"다음 정보를 바탕으로 질문에 답변하세요.\n\n{' '.join(retrieved_chunks)}\n\n질문: {query}"
messages = [{"role": "user", "content": prompt}] # 채팅형식으로 변경
response = client.chat.completions.create(model="gpt-4", messages=messages, max_tokens=100) # chat completions 사용

answer = response.choices[0].message.content.strip()

print(f"질문: {query}")
print(f"답변: {answer}")