from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

import app_model
import chatbot_model
import agent_model

app = FastAPI()

# 정적 페이지를 제공하기 위한 설정을 한다. 
# 이후 static 디렉토리에 파일을 두면 /static/ 밑으로 요청한다.
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

model = app_model.AppModel()
cmodel = chatbot_model.ChatbotModel()
amodel = agent_model.AgentModel()

# URL 부를 때, /say?text=".."
@app.get("/say")
def say_app(text: str = Query()):
    response = model.get_response(text)
    return {"content" : response.content}

# translate로 번역 테스크 만들기
# 언어 선택 가능하도록 모델 변경 ((쿼리 파라미터) 인자가 많을 경우 &을 통해서 연결)
# URL 부를 때, /translate?text=".."&language=".."
@app.get("/translate")
def translate(text: str = Query(), language : str = Query()):
    response = model.get_prompt_response(text, language)
    return {"content" : response.content}

# SSE 기술을 써서 이벤트 스트림으로 내려준다. 클라이언트측 코드는 static/index.html을 참고하자.
# async를 사용하여 SSE 비동기 처리 정상화를 시켜줘야 함. 
@app.get("/says")
def say_app_stream(text: str = Query()):
    async def event_stream():
        async for message in model.get_streaming_response(text):  # 비동기 제너레이터 -> async로 반복해야 함.
            yield f"data: {message.content}\n\n"
    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.get("/chat")
def chat(text: str = Query(), user: str = Query()):
    response = cmodel.get_response(user, 'English', text)
    return {"content" :response.content}

@app.get("/search")
def search(text: str = Query(), user: str = Query()):
    response = amodel.get_response(user, text)
    return {"content" :response.content}
