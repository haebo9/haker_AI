### 기본 세팅
- Python 확장팩
- REST Client 확장팩 설치
- Rainbow CSV 확장팩 설치
- $ bash install.sh 실행

### 파일 설정
- .env : Ai api key 정보를 담은 파일 (반드시 .gitignore에 추가)
    - OPENAI_API_KEY=your_openai_api_key
    - TAVILY_API_KEY=your_tavily_api_key
- .gitignore : git에 업로드 시 경로상에 검색되지 않도록 할 파일을 입력
    - .env
    - __ Pychache __/
- install.sh : 필요한 라이브러리 설치를 위한 파일 
    - $ bash install.sh 실행
- chatmodel.py : 채팅 기능을 담은 파일    
- chatmodel_server.py : FastApi로 chatmodel.py 돌리기
    - $ fastapi dev server.py
- sqlite.db : 챗봇 대화의 내용을 저장하는 db 파일
- test.http : REST Client를 사용하여 서버 GET, POST등의 실행결과 테스트
