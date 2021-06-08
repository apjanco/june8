from fastapi import FastAPI,Request,Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from typing import Optional
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from transformers import pipeline
import requests 

templates = Jinja2Templates(directory="templates")

app = FastAPI()
app.mount("/static", StaticFiles(directory="./static"), name="static")

@app.get("/")
async def read_item(request: Request):
    return templates.TemplateResponse("index.html", {"request": request,})

@app.get("/get_form")
async def read_item(request: Request, question:Optional[str], context:Optional[str]):
    if question or context: 
        message = f"hello {question}."
        return templates.TemplateResponse("get_index.html", {"request": request,"message":message})
    else:
        return templates.TemplateResponse("get_index.html", {"request": request,})


@app.post("/")
async def post_me(
    question: str = Form(...),
    context: str = Form(...),
):
    context = requests.get(context).text
    model_name = "deepset/roberta-base-squad2"

    # a) Get predictions
    nlp = pipeline("question-answering", model=model_name, tokenizer=model_name)
    QA_input = {"question": question, "context": context}
    res = nlp(QA_input)

    # b) Load model & tokenizer
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return res


@app.post("/text_original")
def text(question: str, wikipedia: str):
    context = requests.get(wikipedia).text
    model_name = "deepset/roberta-base-squad2"

    # a) Get predictions
    nlp = pipeline("question-answering", model=model_name, tokenizer=model_name)
    QA_input = {"question": question, "context": context}
    res = nlp(QA_input)

    # b) Load model & tokenizer
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return res
