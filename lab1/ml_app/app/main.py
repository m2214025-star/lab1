from fastapi import FastAPI
from app.schemas import TextRequest, BatchTextRequest, PredictionResponse, ModelInfoResponse
from app.model import classifier, model_name

app = FastAPI(title="Russian Toxicity Classifier API")


#Эндпоинт для проверки сервиса
@app.get("/health")
def health():
    return {"status": "ok"}


#Эндпоинт для предсказания одного текста 
@app.post("/predict", response_model=PredictionResponse)
def predict(request: TextRequest):
    result = classifier(request.text)[0]
    return {"label": result["label"], "score": float(result["score"])}


#Эндпоинт для предсказания списка текстов 
@app.post("/predict_batch", response_model=list[PredictionResponse])
def predict_batch(request: BatchTextRequest):
    results = classifier(request.texts)
    #Преобразуем результат в формат списка словарей с label и score
    return [{"label": r["label"], "score": float(r["score"])} for r in results]


#Эндпоинт с информацией о модели
@app.get("/model_info", response_model=ModelInfoResponse)
def model_info():
    return {"model_name": model_name, "num_labels": len(classifier.model.config.id2label)}
