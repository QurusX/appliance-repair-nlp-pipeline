import re
import torch
from transformers import pipeline
import config


class ApplianceExtractor:
    def __init__(self, model_path=None):
        self.device = 0 if torch.cuda.is_available() else -1
        # Загружаем нашу обученную NER-модель или базовую, если путь не указан
        path = model_path if model_path else config.NER_SAVE_PATH

        try:
            self.ner_pipeline = pipeline(
                "token-classification",
                model=str(path),
                tokenizer=str(path),
                aggregation_strategy="simple",
                device=self.device
            )
        except Exception as e:
            print(f"NER model not found at {path}. System will use Regex only. Error: {e}")
            self.ner_pipeline = None

    def _extract_regex(self, text: str):
        """Извлечение запчастей по паттернам из config.py"""
        return set(re.findall(config.PART_NUMBER_PATTERN, text, re.IGNORECASE))

    def extract_entities(self, text: str):
        results = {
            "model_numbers": set(),
            "part_numbers": set(),
            "error_codes": set()
        }

        # 1. Работаем через NER (если модель загружена)
        if self.ner_pipeline:
            ner_results = self.ner_pipeline(text[:1500])  # Ограничение по длине для BERT
            for ent in ner_results:
                label = ent['entity_group']
                value = ent['word'].strip().upper()

                if label == 'MOD':
                    results["model_numbers"].add(value)
                elif label == 'PRT':
                    results["part_numbers"].add(value)
                elif label == 'ERR':
                    results["error_codes"].add(value)

        # 2. Страховка через Regex (всегда активна)
        regex_parts = self._extract_regex(text)
        results["part_numbers"].update([p.upper() for p in regex_parts])

        # Превращаем сеты в чистые списки для JSON
        return {k: sorted(list(v)) for k, v in results.items()}


if __name__ == "__main__":
    extractor = ApplianceExtractor()
    test_text = "My Whirlpool WFW6620HW0 shows F8 E1. Need part W11165546."
    print(extractor.extract_entities(test_text))