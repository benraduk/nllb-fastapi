from fastapi import FastAPI
from translator import translator
import uvicorn
import json
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import argparse

parser = argparse.ArgumentParser(description="Custom CLI for handling directories.")

# Add arguments for --tokenizer-dir and --ct2-dir
parser.add_argument(
    "--tokenizer-dir", required=True, help="Path to the tokenizer directory."
)
parser.add_argument("--model-dir", required=True, help="Path to the ct2 directory.")

args = parser.parse_args()

# Access the values of the arguments
tokenizer_dir = args.tokenizer_dir
model_dir = args.model_dir

app = FastAPI()
translator = translator(model_dir, tokenizer_dir)


class Request(BaseModel):
    src_lang: str
    tgt_lang: str
    input_text: str


class Error:
    error: str

    def __init__(self, error):
        self.error = error


@app.post("/translate")
async def translate(request: Request):
    invalid_languages = translator.validate_inputs(request.src_lang, request.tgt_lang)
    if len(invalid_languages) > 0:
        error = Error(
            f"Invalid language{'s' if len(invalid_languages) > 1 else '' }: {invalid_languages}, check /languages for a list of valid inputs"
        )
        return JSONResponse(content=jsonable_encoder(error), status_code=400)

    if not translator.check_langs_not_equel(request.src_lang, request.tgt_lang):
        error = Error("Source and target languages must not be the same")
        return JSONResponse(content=jsonable_encoder(error), status_code=400)

    result = translator.translate(**request.__dict__)

    result_json = {
        **request.__dict__,
        "output_text": result,
    }

    return JSONResponse(content=jsonable_encoder(result_json))


@app.get("/languages")
async def langs():
    content = {"languages": translator.lang_list}
    return JSONResponse(content=jsonable_encoder(content))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)