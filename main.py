from typing import Union
from fastapi import FastAPI

# model.py를 가져옴 
import model

# 그 안에 있는 AndModel 클래스의 인스턴스를 생성
and_model = model.AndModel()
or_model = model.OrModel()
not_model = model.NotModel()
xor_model = model.XorModel()

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

# /items/{item_id} 경로
# 중괄호로 닫혀있는 item_id : 경로 매개변수(파라미터)
@app.get("/items/{item_id}")    # endpoint 엔드포인트 
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.get("/predict/{model}/{left}/{right}")    # endpoint 엔드포인트 
def predict(model : str, left: int, right : int):
    if model.lower() == "and":
        result = and_model.predict([left, right])
    elif model.lower() == "or":
        result = or_model.predict([left, right])
    elif model.lower() == "xor":
        result = xor_model.predict([left, right])
    else : 
        return {"result" : "error"}
    return {"model" : model, "input" : [left, right], "result" : result}


@app.get("/predict/not/{num}")
def predict(num : int):
    result = not_model.predict(num)   # not은 입력 하나만 받음
    return {"model" : "not", "input" : num, "result" : result}


@app.post("/train/{model}")
# @app.get("/train")
def train(model : str):
    if model.lower() == "and":
        and_model.train()
    elif model.lower() == "or":
        or_model.train()
    elif model.lower() == "not":
        not_model.train()
    elif model.lower() == "xor":
        xor_model.train()
    else : 
        return {"result" : "error"}
    return {"model" : model, "result": "OK"}

