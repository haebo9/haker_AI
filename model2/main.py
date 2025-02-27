import uvicorn
import Chatmodel_server

app = Chatmodel_server.app

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

# uvicorn main:app --reload

if __name__ == "__main__":
    api_key = "AIzaSyApArsRuO98xud_37C0j9WtJuJqNjwm6f0"  # 실제 API 키로 변경
    model = GeminiModel(api_key)

    text1 = load_json_file('/Users/jaeseoksee/Documents/code/haker_AI/model2/target_data.json')
    text2 = "부루펜은 이부프로펜 성분의 비스테로이드성 항염증제로, 두통, 치통, 생리통 등 다양한 통증 완화 및 해열 효과가 있으며, 성인은 1회 1정씩 필요시 하루 3회까지 복용 가능하나, 위장 장애, 알레르기 반응, 간 기능 이상 등의 부작용과 임산부 및 어린이 복용 시 주의가 필요하고, 다른 약물과의 상호작용 가능성이 있으므로 의사나 약사와 상담 후 복용해야 합니다."
    prompt = f"""
    첫 번째 글:
    {text1}

    두 번째 글:
    {text2}

    찻반쩨 글은 기존에 먹던 약이고, 두번째 글에 있는 새로운 약을 먹을까 한다. 상호작용을 자세하게 분석하여 두 문장 이내로 설명하라.
    반드시 요약하여 핵심적인 문장으로 설명하라.
    """

    response_text = model.generate_content(prompt)
    if response_text:
        print("Generated Response:")
        print(response_text)

    # print("\nStreaming Response:")
    for chunk in model.generate_content_stream(prompt):
        if chunk:
            print(chunk, end="")
    print()