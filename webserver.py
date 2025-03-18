from fastapi import FastAPI, HTTPException, Depends, Request
from translator import translator
import uvicorn
import json
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os

# Read environment variables for model and tokenizer directories
TOKENIZER_DIR = os.getenv("TOKENIZER_DIR", "./nllb-200-distilled-600M")
MODEL_DIR = os.getenv("MODEL_DIR", "./nllb-200-1.3B")
API_KEY = os.getenv("NLLB_API_KEY")

app = FastAPI()
translator = translator(MODEL_DIR, TOKENIZER_DIR)

class TranslationRequest(BaseModel):
    src_lang: str
    tgt_lang: str
    input_text: str

def verify_api_key(request: Request):
    """Middleware function to validate API key"""
    key = request.headers.get("Authorization")
    if not key or key != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Unauthorized")

@app.post("/translate")
async def translate(request: TranslationRequest, api_key: str = Depends(verify_api_key)):
    """Secure translation endpoint"""
    
    invalid_languages = translator.validate_inputs(request.src_lang, request.tgt_lang)
    if invalid_languages:
        return JSONResponse(
            content=jsonable_encoder({"error": f"Invalid languages: {invalid_languages}"}),
            status_code=400
        )

    # Fix function typo (renamed to `check_langs_not_equal`)
    if not translator.check_langs_not_equal(request.src_lang, request.tgt_lang):
        return JSONResponse(
            content=jsonable_encoder({"error": "Source and target languages must not be the same"}),
            status_code=400
        )

    result = translator.translate(request.src_lang, request.tgt_lang, request.input_text)

    result_json = {
        "source_text": request.input_text,
        "source_lang": request.src_lang,
        "target_lang": request.tgt_lang,
        "translated_text": result,
    }

    return JSONResponse(content=jsonable_encoder(result_json))

@app.get("/languages")
async def langs():
    """Returns supported languages"""
    return JSONResponse(content=jsonable_encoder({"languages": translator.lang_list}))

@app.get("/")
async def health_check():
    """Basic health check endpoint"""
    return {"message": "NLLB Translation API is running!"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=6060)
