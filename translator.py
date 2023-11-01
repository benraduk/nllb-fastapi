import ctranslate2
import transformers
from lang_list import lang_list


class translator:
    def __init__(self, model_dir, tokenizer_dir):
        """
        Initialize the translator class with the model and tokenizer directories.
        """
        self.translator = ctranslate2.Translator(model_dir, device="cuda")
        self.lang_list = lang_list

    def translate(self, src_lang, tgt_lang, input_text):
        """
        Translate the input text from the source language to the target language.
        """

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "./nllb-200-distilled-600M", src_lang=src_lang
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
