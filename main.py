from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import MarianMTModel, MarianTokenizer
from langdetect import detect

app = FastAPI()

# CORS (for React / frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request schema
class TranslateRequest(BaseModel):
    text: str
    target_lang: str  # "en", "hi", "fr"

# Supported translation models
MODEL_MAP = {
    ("en", "hi"): "Helsinki-NLP/opus-mt-en-hi",
    ("hi", "en"): "Helsinki-NLP/opus-mt-hi-en",
    ("en", "fr"): "Helsinki-NLP/opus-mt-en-fr",
    ("fr", "en"): "Helsinki-NLP/opus-mt-fr-en",
}

# Cache loaded models (important)
loaded_models = {}

def get_model(src, tgt):
    key = (src, tgt)
    if key not in MODEL_MAP:
        raise HTTPException(
            status_code=400,
            detail=f"Translation from {src} to {tgt} not supported"
        )

    if key not in loaded_models:
        model_name = MODEL_MAP[key]
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        loaded_models[key] = (tokenizer, model)

    return loaded_models[key]

@app.post("/translate")
def translate(req: TranslateRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Empty text")

    tgt_lang = req.target_lang.lower()

    # Force src_lang if text contains Hindi characters
    if any("\u0900" <= c <= "\u097F" for c in req.text):
        src_lang = "hi"
    elif any("\u0600" <= c <= "\u06FF" for c in req.text):
        src_lang = "ar"  # example for Arabic
    elif any("A" <= c <= "z" for c in req.text):
        src_lang = "en"
    else:
        # fallback to detect
        src_lang = detect(req.text)

    # normalize
    if src_lang.startswith("hi"):
        src_lang = "hi"
    elif src_lang.startswith("fr"):
        src_lang = "fr"
    else:
        src_lang = "en"

    try:
        tokenizer, model = get_model(src_lang, tgt_lang)
    except HTTPException as e:
        return {"error": str(e.detail)}

    inputs = tokenizer(req.text, return_tensors="pt", padding=True)
    outputs = model.generate(**inputs, max_length=512)
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {
        "source_lang": src_lang,
        "target_lang": tgt_lang,
        "translated_text": translated_text
    }
