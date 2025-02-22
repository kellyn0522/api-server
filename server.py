from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

import app_model

app = FastAPI()

model = app_model.AppModel()

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