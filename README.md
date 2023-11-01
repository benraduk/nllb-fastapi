# Prerequisits

This was written in python-3.10 and might now work with older versions of python.

# Step by step to getting the server to work

1. Install python requirements `pip install -r requirements.txt`
2. Change model to ct2 format `ct2-transformers-converter --model <model_dir> --output_dir=<output_dir>`
3. Start the server by running `python webserver.py --tokenizer-dir <tokenizer_dir> --model-dir <ct2_model_dir>`

Notice that the tokenizer dir is the regular model directory (as gotten from huggingface), left only with the tokenizer files:

```
config.json
special_tokens_map.json
tokenizer_config.json
tokenizer.json
```
