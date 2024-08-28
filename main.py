from fastapi import FastAPI
from transformers import pipeline

app = FastAPI()

@app.get("/")
async def root():
    return {"messsage" : "Successfully Initiated"}

# 유저로부터 text를 받아서 감정 분석 결과를 반환해주는 API
@app.get("/sentiment/")
async def sentiment(text: str = None):
    classifier = pipeline("text-classification", model="hun3359/klue-bert-base-sentiment")
    preds = classifier(text, top_k=None)

    sorted_preds = sorted(preds, key=lambda x: x['score'], reverse=True)
    
    for item in sorted_preds:
        item['score'] = round(item['score'], 5)
    
    return sorted_preds