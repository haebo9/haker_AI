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

# itemName 임베딩 생성 및 데이터 프레임에 추가
item_names = df['itemName'].tolist()
item_name_embeddings = []

for item_name in item_names:
    response = client.embeddings.create(input=item_name, model="text-embedding-ada-002")
    item_name_embeddings.append(response.data[0].embedding)

df['item_name_embedding'] = item_name_embeddings
df['item_name_embedding'] = df['item_name_embedding'].apply(np.array)

# 질문에서 약 이름 추출 및 임베딩
query = "활명수의 부작용이 있어?"
query_embedding = client.embeddings.create(input=query, model="text-embedding-ada-002").data[0].embedding

# 유사도 계산
similarities = cosine_similarity(np.array(query_embedding).reshape(1, -1), np.array(df['item_name_embedding'].tolist()))
most_similar_index = np.argmax(similarities)
most_similar_item = item_names[most_similar_index]

# 해당 약에 대한 행 추출 및 출력
selected_row = df[df['itemName'] == most_similar_item]

if not selected_row.empty:
    print(selected_row)

    # LLM에 검색된 청크 입력 및 답변 생성
    extracted_info = selected_row.iloc[0]['efcyQesitm'] + " " + selected_row.iloc[0]['useMethodQesitm'] + " " + selected_row.iloc[0]['atpnQesitm'] + " " + selected_row.iloc[0]['intrcQesitm'] + " " + selected_row.iloc[0]['seQesitm']
    print(extracted_info)
    prompt = f"제공한 정보를 우선적으로 참고하여 질문에 대한 답변을 추출하라. 제공한 정보는 반드시 {query}에 대한 설명이다. \n\n정보: {extracted_info}\n\n질문: {query}"
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(model="gpt-4o-mini", messages=messages, max_tokens=500)
    answer = response.choices[0].message.content.strip()

    print(f"질문: {query}")
    print(f"답변: {answer}")
else:
    print("제공된 정보에서 해당 약을 찾을 수 없습니다.")