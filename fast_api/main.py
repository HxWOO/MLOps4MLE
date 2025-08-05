# main.py, openAPI를 만들기 위해 fastAPI를 import 햇음 
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Create a FastAPI instance, 여기 생성하는 객체의 변수명에 따라 실행할때 쓴 명령어가 달라짐 현재는 fast_api.main:app
app = FastAPI()

# User database
USER_DB = {}

# Fail response
NAME_NOT_FOUND = HTTPException(status_code=400, detail="Name not found.")

# path를 설정해줌, Http get요청이 오면 read_root api 실행
@app.get("/")
def read_root():
    return {"Hello": "World"}