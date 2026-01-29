from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import MarianMTModel, MarianTokenizer
from langdetect import detect

app = FastAPI()

# CORS
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

# Cache for loaded models
loaded_models = {}

def get_model(src, tgt):
    """Load MarianMT model and tokenizer, with error handling."""
    key = (src, tgt)
    if key not in MODEL_MAP:
        return None
    if key not in loaded_models:
        try:
            model_name = MODEL_MAP[key]
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)
            loaded_models[key] = (tokenizer, model)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load model {model_name}: {e}")
    return loaded_models[key]

def translate_text(text, src, tgt):
    """Translate text, using English pivot if direct model not available."""
    model_data = get_model(src, tgt)
    if model_data:
        tokenizer, model = model_data
        inputs = tokenizer(text, return_tensors="pt", padding=True)
        outputs = model.generate(**inputs, max_length=512)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Pivot through English if no direct model
    if src != "en" and tgt != "en":
        intermediate = translate_text(text, src, "en")
        return translate_text(intermediate, "en", tgt)
    
    raise HTTPException(status_code=400, detail=f"No model available for {src} → {tgt}")

@app.post("/translate")
def translate(req: TranslateRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty text")

    tgt_lang = req.target_lang.lower()
    if tgt_lang not in ["en", "hi", "fr"]:
        raise HTTPException(status_code=400, detail=f"Unsupported target language: {tgt_lang}")

    # Detect source language for known languages
    if any("\u0900" <= c <= "\u097F" for c in text):
        src_lang = "hi"
    elif any("A" <= c <= "Z" or "a" <= c <= "z" for c in text):
        # Could be English or French; use langdetect
        detected = detect(text)
        if detected.startswith("fr"):
            src_lang = "fr"
        else:
            src_lang = "en"
    else:
        detected = detect(text)
        if detected.startswith("hi"):
            src_lang = "hi"
        elif detected.startswith("fr"):
            src_lang = "fr"
        else:
            src_lang = "en"

    try:
        translated_text = translate_text(text, src_lang, tgt_lang)
    except HTTPException as e:
        return {"error": str(e.detail)}

    return {
        "source_lang": src_lang,
        "target_lang": tgt_lang,
        "translated_text": translated_text
    }
