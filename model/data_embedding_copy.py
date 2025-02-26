import pandas as pd
from openai import OpenAI
import numpy as np
from dotenv import load_dotenv
import os
from sklearn.metrics.pairwise import cosine_similarity

# OpenAI 클라이언트 초기화
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 데이터 프레임 csv 파일 읽어오기
df = pd.read_csv('/workspaces/haker_AI/model/drug_data.csv')

# NaN 값을 빈 문자열로 대체
df['efcyQesitm'] = df['efcyQesitm'].fillna('')
df['useMethodQesitm'] = df['useMethodQesitm'].fillna('')
df['atpnQesitm'] = df['atpnQesitm'].fillna('')
df['intrcQesitm'] = df['intrcQesitm'].fillna('')
df['seQesitm'] = df['seQesitm'].fillna('')

# combined_text 열 생성
df['combined_text'] = df['efcyQesitm'] + " " + df['useMethodQesitm'] + " " + df['atpnQesitm'] + " " + df['intrcQesitm'] + " " + df['seQesitm']

# 임베딩 생성 및 데이터 프레임에 추가
embeddings = []
for index, row in df.iterrows():
    response = client.embeddings.create(input=row['combined_text'], model="text-embedding-ada-002")
    embeddings.append(response.data[0].embedding)

df['ada_embedding'] = embeddings
df['ada_embedding'] = df['ada_embedding'].apply(np.array)

print(df)

# 벡터 데이터베이스 저장 (예시)
vector_db = dict(zip(df['combined_text'].tolist(), df['ada_embedding'].tolist()))

# 사용자 질문 임베딩
query = "활명수"
query_embedding = client.embeddings.create(input=query, model="text-embedding-ada-002").data[0].embedding

# 유사도 계산 및 검색
similarities = []
for chunk, embedding in vector_db.items():
    similarity = cosine_similarity(np.array(query_embedding).reshape(1, -1), np.array(embedding).reshape(1, -1))[0][0]
    similarities.append((similarity, chunk))

similarities.sort(reverse=True)
retrieved_chunks = [chunk for _, chunk in similarities[:2]]
print("관련성이 높은 정보 top3: ", retrieved_chunks)

# 언어 모델(LLM)에 검색된 청크 입력 및 답변 생성 (예시)
prompt = f"다음 정보를 바탕으로 질문에 답변하세요.\n\n{' '.join(retrieved_chunks)}\n\n질문: {query}"
messages = [{"role": "user", "content": prompt}]
response = client.chat.completions.create(model="gpt-4", messages=messages, max_tokens=1000)

answer = response.choices[0].message.content.strip()

print(f"질문: {query}")
print(f"답변: {answer}")