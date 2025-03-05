import google.generativeai as genai
import json

class GeminiModel:
    def __init__(self, api_key: str, ROOT_PATH : str, model_name: str = "gemini-2.0-flash"): # 모델이름을 확인후 수정하세요.
        self.api_key = api_key
        self.model_name = model_name
        self.ROOT_PATH = ROOT_PATH
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)
        with open(f"{ROOT_PATH}/data/target_data.json", 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        self.json_data = json_data.keys() # JSON 데이터를 저장할 딕셔너리

    def generate_content(self, prompt: str) -> str:
        try:
            # JSON 데이터의 키 값에 해당하는 경우 요약
            if prompt in self.json_data:
                summary_prompt = f"문장을 이해하기 가능한 이해하기 쉽게 6개의 문장 이내로 되도록 하고 한글로 답하라. 해당 약의 효능효과와 주의사항을 중점적으로 설명하라. : {self.json_data[prompt]}"
                response = self.model.generate_content(summary_prompt)
                return response.text

            # 새로운 값이 입력된 경우 상관관계 분석
            else:
                if self.json_data:  # 기존 데이터가 있는 경우에만 분석
                    analysis_prompt = f"문장을 이해하기 가능한 이해하기 쉽게 6개의 문장 이내로 되도록 하고 한글로 답하라. 해당 약의 효능효과와 주의사항 그리고 Analyze the relationship between the following new data: '{prompt}' and the existing data: '{self.json_data}'"
                    response = self.model.generate_content(analysis_prompt)
                    return response.text
                else:
                    return "No existing data to analyze. Please load JSON data first."

        except Exception as e:
            print(f"Error generating content: {e}")
            return None

    def generate_content_stream(self, prompt: str):
        try:
            response = self.model.generate_content(prompt, stream=True)
            for chunk in response:
                yield chunk.text
        except Exception as e:
            print(f"Error generating content stream: {e}")
            yield f"Error: {e}"

    def load_json_file(self, filepath: str):
        """Loads a JSON file and returns its contents."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                self.json_data = json.load(f)  # 클래스 변수에 JSON 데이터 저장
                return self.json_data
        except FileNotFoundError:
            print(f"Error: File not found at {filepath}")
            return None
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in {filepath}")
            return None