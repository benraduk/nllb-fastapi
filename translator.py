import ctranslate2
import transformers
from lang_list import lang_list


class translator:
    def __init__(self, model_dir, tokenizer_dir):
        """
        Initialize the translator class with the model and tokenizer directories.
        """
        try:
            # Try to initialize with CUDA first
            self.translator = ctranslate2.Translator(model_dir, device="cuda")
            print("Using CUDA for translation")
        except RuntimeError as e:
            # Fall back to CPU if CUDA is not available
            print(f"CUDA initialization failed: {str(e)}")
            print("Falling back to CPU for translation")
            self.translator = ctranslate2.Translator(model_dir, device="cpu")
            
        self.lang_list = lang_list
        self.tokenizer_dir = tokenizer_dir

    def translate(self, src_lang, tgt_lang, input_text):
        """
        Translate the input text from the source language to the target language.
        """

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.tokenizer_dir, src_lang=src_lang
        )

        input_tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(input_text))

        target_prefix = [tgt_lang]

        results = self.translator.translate_batch(
            [input_tokens], target_prefix=[target_prefix]
        )
        target = results[0].hypotheses[0][1:]

        output = tokenizer.decode(tokenizer.convert_tokens_to_ids(target))

        return output

    def validate_inputs(self, src_lang, tgt_lang):
        """
        Validate the source and target languages.
        """
        invalid_languages = []
        if src_lang not in self.lang_list:
            invalid_languages.append(src_lang)

        if tgt_lang not in self.lang_list:
            invalid_languages.append(tgt_lang)

        return invalid_languages

    def check_langs_not_equal(self, src_lang, tgt_lang):
        """
        Check if the source and target languages are not the same.
        """

        if src_lang == tgt_lang:
            return False
        return True
