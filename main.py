from fastapi import FastAPI, File, UploadFile, Form, Request
from transformers import pipeline
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import codecs


templates = Jinja2Templates(directory='htmldirectory')


class Item(BaseModel):
    text: str


app = FastAPI()
classifier = pipeline("sentiment-analysis")


@app.get("/", response_class=HTMLResponse)
def root(request: Request):
    return templates.TemplateResponse('home.html',{'request': request})


def predict(df: str):
    return classifier(df)[0]


@app.post("/",response_class=HTMLResponse)
async def handle_form(request: Request, assignment_file: UploadFile = File(...)):
    content_assighment = await assignment_file.read()
    content_assighment = codecs.decode(content_assighment, 'UTF-8')
    return templates.TemplateResponse('home.html',{'request': request, 'result': predict(content_assighment)})


