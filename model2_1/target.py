import pandas as pd

# read_csv file
df = pd.read_csv('/workspaces/haker_AI/model2/sample_data.csv')
target = pd.read_json("/workspaces/haker_AI/model2/input_med.json")["key"].tolist() # 리스트 형태
target_df = df[df['제품명'].isin(target)]
df_indexed = target_df.set_index('제품명')
df_indexed.to_json("/workspaces/haker_AI/model2/target_data.json", orient='index', force_ascii=False, indent=4)