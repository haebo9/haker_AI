import pandas as pd
from openai import OpenAI
import numpy as np
from dotenv import load_dotenv
import os
import pickle
import time

start_time = time.time()

# OpenAI 클라이언트 초기화
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 데이터 프레임 csv 파일 읽어오기
df = pd.read_csv('/workspaces/haker_AI/model/drug500_data.csv')

# NaN 값을 빈 문자열로 대체
# NaN 값을 빈 문자열로 대체
df['효능효과'] = df['효능효과'].fillna('')
df['용법용량'] = df['용법용량'].fillna('')
df['주의사항'] = df['주의사항'].fillna('')
df['상호작용'] = df['상호작용'].fillna('')
df['부작용'] = df['부작용'].fillna('')

# itemName 임베딩 생성 및 데이터 프레임에 추가
item_names = df['제품명'].tolist()
item_name_embeddings = []

for item_name in item_names:
    response = client.embeddings.create(input=item_name, model="text-embedding-ada-002")
    item_name_embeddings.append(response.data[0].embedding)

df['제품명_embedding'] = item_name_embeddings
df['제품명_embedding'] = df['제품명_embedding'].apply(np.array)

# 임베딩 결과를 pickle 파일로 저장
with open('drug_embeddings.pkl', 'wb') as f:
    pickle.dump(df, f)

print("임베딩 생성 및 저장 완료.")

# 종료 시간 기록 및 실행 시간 계산
end_time = time.time()
execution_time = end_time - start_time

print(f"실행 시간: {execution_time:.2f} 초")