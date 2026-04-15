import pandas as pd
import torch
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM
)
from src.extractor import ApplianceExtractor
import config


class ApplianceRepairPipeline:
    def __init__(self):
        self.device = config.DEVICE
        print(f"Initializing pipeline on {self.device}...")

        # 1. Загрузка классификатора
        self.clf_tokenizer = AutoTokenizer.from_pretrained(config.CLASSIFIER_SAVE_PATH)
        self.clf_model = AutoModelForSequenceClassification.from_pretrained(
            config.CLASSIFIER_SAVE_PATH
        ).to(self.device)

        # 2. Загрузка генератора
        self.gen_tokenizer = AutoTokenizer.from_pretrained(config.GENERATOR_SAVE_PATH)
        self.gen_model = AutoModelForSeq2SeqLM.from_pretrained(
            config.GENERATOR_SAVE_PATH
        ).to(self.device)

        # 3. Инициализация экстрактора
        self.extractor = ApplianceExtractor()

    def process_record(self, question: str, answer: str):
        full_text = f"QUESTION: {question}\nANSWER: {answer}"

        # --- Шаг 1: Классификация ---
        clf_inputs = self.clf_tokenizer(
            full_text, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)

        with torch.no_grad():
            clf_outputs = self.clf_model(**clf_inputs)
            probs = torch.sigmoid(clf_outputs.logits)[0]
            is_repair = probs[0].item() > 0.5
            has_solution = probs[1].item() > 0.5

        if not is_repair:
            return {
                "is_repair": False,
                "category": "Non-repair context",
                "has_solution": False,
                "entities": {}
            }

        # --- Шаг 2: Генерация категории ---
        gen_prompt = f"categorize appliance repair problem: {full_text}"
        gen_inputs = self.gen_tokenizer(
            gen_prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)

        with torch.no_grad():
            gen_outputs = self.gen_model.generate(**gen_inputs, max_new_tokens=30)
            category = self.gen_tokenizer.decode(gen_outputs[0], skip_special_tokens=True)

        # --- Шаг 3: Извлечение сущностей ---
        entities = self.extractor.extract_entities(full_text)

        return {
            "is_repair": True,
            "category": category,
            "has_solution": has_solution,
            "entities": entities
        }

    def process_file(self, input_path: str, output_path: str, batch_size: int = 32):
        df = pd.read_excel(input_path)

        results = []
        print(f"Processing {len(df)} rows in batches of {batch_size}...")

        for i in tqdm(range(0, len(df), batch_size)):
            batch_df = df.iloc[i: i + batch_size]
            questions = batch_df['Question'].fillna("").astype(str).tolist()
            answers = batch_df['Answer'].fillna("").astype(str).tolist()
            full_texts = [f"QUESTION: {q}\nANSWER: {a}" for q, a in zip(questions, answers)]

            # 1. Пакетная классификация
            clf_inputs = self.clf_tokenizer(
                full_texts, return_tensors="pt", truncation=True, padding=True, max_length=512
            ).to(self.device)

            with torch.no_grad():
                clf_outputs = self.clf_model(**clf_inputs)
                probs = torch.sigmoid(clf_outputs.logits)
                is_repair_batch = (probs[:, 0] > 0.5).cpu().tolist()
                has_solution_batch = (probs[:, 1] > 0.5).cpu().tolist()

            repair_indices = [idx for idx, val in enumerate(is_repair_batch) if val]
            categories = ["Non-repair context"] * len(full_texts)

            if repair_indices:
                repair_texts = [full_texts[idx] for idx in repair_indices]
                gen_prompts = [f"summarize repair: {t}" for t in repair_texts]


                gen_inputs = self.gen_tokenizer(
                    gen_prompts, return_tensors="pt", truncation=True, padding=True, max_length=512
                ).to(self.device)

                with torch.no_grad():
                    gen_outputs = self.gen_model.generate(**gen_inputs, max_new_tokens=30)
                    decoded_cats = self.gen_tokenizer.batch_decode(gen_outputs, skip_special_tokens=True)

                for idx, cat in zip(repair_indices, decoded_cats):
                    categories[idx] = cat

            for idx in range(len(full_texts)):
                if is_repair_batch[idx]:
                    ents = self.extractor.extract_entities(full_texts[idx])
                else:
                    ents = {"model_numbers": [], "part_numbers": [], "error_codes": []}

                results.append({
                    "Question": questions[idx],
                    "Answer": answers[idx],
                    "Is_Repair": is_repair_batch[idx],
                    "Category": categories[idx],
                    "Has_Solution": has_solution_batch[idx],
                    "Model_Numbers": ", ".join(ents.get("model_numbers", [])),
                    "Part_Numbers": ", ".join(ents.get("part_numbers", [])),
                    "Error_Codes": ", ".join(ents.get("error_codes", []))
                })

        output_df = pd.DataFrame(results)
        output_df.to_excel(output_path, index=False)
        print(f"\nDone! Result saved to {output_path}")

if __name__ == "__main__":
    # pipeline = ApplianceRepairPipeline()
    # pipeline.process_file("data/raw/input.xlsx", "data/processed/output.xlsx")
    pass