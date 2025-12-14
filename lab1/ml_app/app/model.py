# app/model.py
from transformers import pipeline

# Название модели Hugging Face
model_name = "s-nlp/russian_toxicity_classifier"

# Загружаем классификатор
classifier = pipeline("text-classification", model=model_name)

