# app/utils.py
import json
from langdetect import detect
from deep_translator import GoogleTranslator

def detect_language(text: str) -> str:
    try:
        lang = detect(text)
        return lang
    except:
        return "en"

def translate_text(text: str, target: str="en"):
    try:
        if target == "en":
            return GoogleTranslator(source='auto', target='en').translate(text)
        else:
            return GoogleTranslator(source='auto', target=target).translate(text)
    except Exception as e:
        # fallback: return original
        return text

def load_json(path):
    with open(path, 'r', encoding='utf8') as f:
        return json.load(f)
