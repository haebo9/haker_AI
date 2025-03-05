import pandas as pd
import os
import json

class DataLoader: 
    def __init__(self, root_path):
        self.root_path = root_path
        self.df = None  # 데이터 프레임 (sample_data.csv)
        self.target = None  # 필터링할 대상 리스트 (input_med.json)
        self.target_data_dict = None  # JSON으로 저장될 데이터
        self.new_data = None  # 새로운 JSON 데이터 (new_data.json)
        self._load_and_prepare_data() # 인스턴스 생성 시 데이터 로드
    
    def _load_and_prepare_data(self):
        self.load_data()
        self.prepare_data()
        self.load_new_data()

    # sample_data.csv와 input_med.json 파일을 로드하여 데이터 프레임과 타겟 리스트를 생성
    def load_data(self):
        try:
            self.df = pd.read_csv(os.path.join(self.root_path, "sample_data.csv"))
            with open(os.path.join(self.root_path, "input_med.json"), 'r', encoding='utf-8') as f:
                self.target = json.load(f)["key"]
            print("[INFO] 데이터 로드 완료.")
        except Exception as e:
            print(f"[ERROR] 데이터 로드 중 오류 발생: {e}")
            raise

    # '제품명' 기준으로 데이터 필터링 후 JSON 파일로 저장
    def prepare_data(self): 
        try:
            # '제품명' 기준 필터링
            target_df = self.df[self.df['제품명'].isin(self.target)]
            
            # 인덱스 설정 후 JSON 저장
            df_indexed = target_df.set_index('제품명')
            target_data_path = os.path.join(self.root_path, "target_data.json")
            df_indexed.to_json(target_data_path, orient='index', force_ascii=False, indent=4)

            # 저장된 JSON 파일 다시 로드
            with open(target_data_path, 'r', encoding='utf-8') as f:
                self.target_data_dict = json.load(f)

            print(f"[INFO] 필터링된 데이터 저장 완료: {target_data_path}")
        except Exception as e:
            print(f"[ERROR] 데이터 필터링 및 저장 중 오류 발생: {e}")
            raise

    # new_data.json 파일 로드
    def load_new_data(self):
        try:
            new_data_path = os.path.join(self.root_path, "new_data.json")
            with open(new_data_path, "r", encoding="utf-8") as f:
                self.new_data = json.load(f)
            print("[INFO] 새로운 데이터 로드 완료.")
        except Exception as e:
            print(f"[ERROR] 새로운 데이터 로드 중 오류 발생: {e}")
            raise

    # 전체 데이터 처리 실행 함수
    def run(self):
        self.load_data()
        self.prepare_data()
        self.load_new_data()
        return self.df, self.target, self.target_data_dict, self.new_data
    
    def __call__(self, root_path=None):
        """
        객체를 호출 가능하게 만들어, 데이터를 반환합니다.
        root_path를 인자로 받을 수 있도록 수정하여,
        필요한 경우 다른 경로의 데이터를 로드할 수 있습니다.
        """
        if root_path: # 새로운 root_path가 제공되면, 해당 경로로 다시 로드합니다.
            self.__init__(root_path)
        return self.df, self.target, self.target_data_dict, self.new_data