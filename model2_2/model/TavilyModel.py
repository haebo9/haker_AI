from tavily import TavilyClient
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import json
import re  # 정규 표현식 모듈 import


class TavilyModel:
    def __init__(self, ROOT_PATH):
        load_dotenv(os.path.join(ROOT_PATH, 'utils', '.env'))
        self.tavily_client = TavilyClient(os.getenv("TAVILY_API_KEY"))
        self.gpt_4o = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini", temperature=0.1)
        self.data_file = os.path.join(ROOT_PATH, 'data', 'new_data.json')
        self.load_data()

    def load_data(self):
        """JSON 파일 로드 (파일이 반드시 존재한다고 가정)"""
        try:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:  # 두 예외를 함께 처리
            print(f"[WARNING] new_data.json 파일 로드 실패: {e}. 파일 초기화.")
            self.data = {}
            if not os.path.exists(os.path.dirname(self.data_file)): #디렉토리가 존재하지 않을시 생성.
                os.makedirs(os.path.dirname(self.data_file))
            with open(self.data_file, 'w', encoding='utf-8') as f:  # 빈 파일 생성
                json.dump(self.data, f)


    def get_medication_info(self, medication_name):
        """Tavily API를 사용하여 약 정보 검색"""
        try:
            search_response = self.tavily_client.search(
                query=f"{medication_name} 복약 정보",
                search_type="advanced",
                # time_range="month" #월간 검색은 최신성이 떨어질 수 있음.
            )

            results = search_response.get("results", [])

            if not results:
                return f"'{medication_name}'에 대한 최신 복약 정보를 찾을 수 없습니다."

            response_text = f" **'{medication_name}' 복약 정보**\n\n"
            for result in results[:5]:
                response_text += f" **{result.get('title', '제목 없음')}**\n"
                response_text += f" [출처]({result.get('url', '#')})\n"
                response_text += f" {result.get('content', '요약 정보 없음.')}\n\n"

            return response_text

        except Exception as e:
            return f"오류 발생: {str(e)}"

    def generate_chatbot_response(self, medication_name):
        """OpenAI API를 사용하여 챗봇 응답 생성"""
        info = self.get_medication_info(medication_name)

        prompt = f"""
        복약 정보를 쉽고 친절하게 알려드립니다.
        '{medication_name}'에 대해 궁금하신 내용을 아래 정보를 바탕으로 알려주세요.

        {info}

        다음 형식으로 답변해주세요:
        효과/효능: ...
        복용법: ...
        주의사항: ...

        정보가 없는 경우 'None'이라고 답변해주세요.
        """

        response = self.gpt_4o.invoke(prompt)
        return response.content

    def update_data(self, medication_name, data):
        """JSON 파일 업데이트"""
        self.data[medication_name] = data
        with open(self.data_file, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=4)
        print(f"[INFO] '{medication_name}' 정보가 new_data.json 파일에 추가/업데이트되었습니다.")

    def extract_info(self, response_text):
        """응답 텍스트에서 효과/효능, 복용법, 주의사항 추출"""
        info_dict = {}

        # 정규 표현식 패턴 정의 (각 섹션의 시작과 끝을 명확히)
        effect_pattern = r"효과/효능:\s*(.*?)(?:\n복용법:|\n주의사항:|$)"
        dosage_pattern = r"복용법:\s*(.*?)(?:\n효과/효능:|\n주의사항:|$)"
        precautions_pattern = r"주의사항:\s*(.*?)$"

        # 각 패턴에 대해 매칭 수행 및 결과 저장
        effect_match = re.search(effect_pattern, response_text, re.DOTALL)
        dosage_match = re.search(dosage_pattern, response_text, re.DOTALL)
        precautions_match = re.search(precautions_pattern, response_text, re.DOTALL)

        # 매칭된 결과가 있으면 해당 텍스트 추출, 없으면 None 할당
        info_dict["효과/효능"] = effect_match.group(1).strip() if effect_match else None
        info_dict["복용법"] = dosage_match.group(1).strip() if dosage_match else None
        info_dict["주의사항"] = precautions_match.group(1).strip() if precautions_match else None

        return info_dict


    def __call__(self, medication_name):
        """약 이름으로 정보 조회 및 업데이트"""
        if medication_name in self.data:
            return self.data[medication_name]  # 이미 있으면 기존 정보 반환
        else:
            print(f"[INFO] '{medication_name}' 정보가 new_data.json 파일에 없습니다. 검색을 시작합니다.")
            response_text = self.generate_chatbot_response(medication_name)
            print(f"[DEBUG] Raw response from generate_chatbot_response:\n{response_text}")
            extracted_info = self.extract_info(response_text)  # 정보 추출

            if extracted_info:  # 빈 딕셔너리가 아니라면 (정보가 있다면)
                self.update_data(medication_name, extracted_info)
                return extracted_info  # 새로 검색한 정보 반환
            else:  # 정보 추출에 실패한 경우.
                return {"error": f"'{medication_name}'에 대한 정보를 추출하는데 실패했습니다."}


# 테스트를 위한 코드
if __name__ == "__main__":
    # 1.  '.env' 파일이 있는  루트 디렉토리 설정 (utils의 상위 디렉토리)
    ROOT_PATH = "/Users/jaeseoksee/Documents/code/haker_AI/model2"  # 실제 프로젝트 루트 경로로 변경!

    # 2. TavilyModel 인스턴스 생성
    chatbot = TavilyModel(ROOT_PATH=ROOT_PATH)

    # 3. "부루펜" 검색
    print("--- 부루펜 검색 ---")
    buru_result = chatbot("부루펜")
    print(buru_result)

    # 4. "활명수" 검색
    print("\n--- 활명수 검색 ---")
    hwal_result = chatbot("활명수")
    print(hwal_result)