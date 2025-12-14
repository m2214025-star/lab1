from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


#Тест эндпоинта /health
def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


#Тест эндпоинта /predict (один текст)
def test_predict():
    data = {"text": "Ты такой милый!"}
    response = client.post("/predict", json=data)
    assert response.status_code == 200
    json_data = response.json()
    assert "label" in json_data
    assert "score" in json_data
    assert isinstance(json_data["label"], str)
    assert isinstance(json_data["score"], float)


#Тест эндпоинта /predict_batch (список текстов)
def test_predict_batch():
    data = {"texts": ["Ты хороший человек", "Это ужасно!"]}
    response = client.post("/predict_batch", json=data)
    assert response.status_code == 200
    json_data = response.json()
    assert isinstance(json_data, list)
    assert len(json_data) == 2
    for item in json_data:
        assert "label" in item
        assert "score" in item


#Тест эндпоинта /model_info
def test_model_info():
    response = client.get("/model_info")
    assert response.status_code == 200
    json_data = response.json()
    assert "model_name" in json_data
    assert "num_labels" in json_data


#Тест на валидацию ошибок (например, пустой текст)
def test_predict_validation_error():
    data = {"text": ""}
    response = client.post("/predict", json=data)
    assert response.status_code == 422  # FastAPI + Pydantic автоматически проверяют схемы

