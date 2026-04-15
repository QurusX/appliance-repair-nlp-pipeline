import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer, DataCollatorForTokenClassification
)
import config


class ApplianceTrainer:
    def __init__(self):
        self.device = config.DEVICE
        self.labeled_data = config.LABELED_DATA
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    def _load_data(self):
        import pandas as pd
        from datasets import Dataset

        print("Merging Excel and JSONL data...")
        # 1. Загружаем разметку из JSONL
        labels_df = pd.read_json(self.labeled_data, lines=True)

        # 2. Загружаем исходный текст из Excel
        raw_df = pd.read_excel(config.INPUT_EXCEL)
        raw_df['original_row_index'] = raw_df.index

        # 3. Объединяем по индексу строки
        merged_df = pd.merge(
            labels_df,
            raw_df[['original_row_index', 'Question', 'Answer']],
            on='original_row_index'
        )

        # Создаем колонку 'text', которую просят модели
        merged_df['text'] = "QUESTION: " + merged_df['Question'].astype(str) + "\nANSWER: " + merged_df[
            'Answer'].astype(str)

        # Превращаем в формат Hugging Face
        dataset = Dataset.from_pandas(merged_df)
        return dataset.train_test_split(test_size=0.1, seed=42)

    def train_classifier(self):
        print("--- Training Classifier (DistilBERT) ---")
        raw_datasets = self._load_data()
        model_id = "distilbert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        def preprocess(examples):
            # ТЕПЕРЬ УЧИМ НА ТЕКСТЕ (text), а не на category
            tokenized = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
            labels = [[float(r), float(s)] for r, s in zip(examples["is_repair_related"], examples["has_solution"])]
            tokenized["labels"] = labels
            return tokenized

        ds = raw_datasets.map(preprocess, batched=True, remove_columns=raw_datasets["train"].column_names)

        model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            num_labels=2,
            problem_type="multi_label_classification"
        ).to(self.device)

        args = TrainingArguments(
            output_dir=str(config.CLASSIFIER_SAVE_PATH),
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=config.BATCH_SIZE,
            per_device_eval_batch_size=config.BATCH_SIZE,
            num_train_epochs=5,
            weight_decay=0.01,
            fp16=torch.cuda.is_available(),
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss"
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=ds["train"],
            eval_dataset=ds["test"]
        )

        trainer.train()

        model.save_pretrained(config.CLASSIFIER_SAVE_PATH)
        tokenizer.save_pretrained(config.CLASSIFIER_SAVE_PATH)
        print(f"Classifier saved to {config.CLASSIFIER_SAVE_PATH}")

    def train_generator(self):
        print("\n--- Training Generator (T5) ---")
        raw_datasets = self._load_data()
        model_id = "t5-small"
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        def preprocess(examples):
            # ОШИБКА БЫЛА ТУТ: На вход подаем ТЕКСТ (вопрос+ответ), а не категорию
            inputs = [f"summarize repair: {t}" for t in examples["text"]]
            model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")

            # На выход подаем КАТЕГОРИЮ, которую сгенерировал Gemini
            labels = tokenizer(text_target=examples["category"], max_length=64, truncation=True, padding="max_length")

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        ds = raw_datasets.map(preprocess, batched=True, remove_columns=raw_datasets["train"].column_names)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(self.device)

        args = TrainingArguments(
            output_dir=str(config.GENERATOR_SAVE_PATH),
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=5e-4,
            per_device_train_batch_size=config.BATCH_SIZE,
            num_train_epochs=15,
            fp16=torch.cuda.is_available(),
            load_best_model_at_end=True
        )

        trainer = Trainer(
            model=model, args=args, train_dataset=ds["train"], eval_dataset=ds["test"]
        )
        trainer.train()
        model.save_pretrained(config.GENERATOR_SAVE_PATH)
        tokenizer.save_pretrained(config.GENERATOR_SAVE_PATH)

    def train_ner(self):
        print("\n--- Training NER Extractor (DistilBERT-NER) ---")
        raw_datasets = self._load_data()
        model_id = "distilbert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Мэппинг тегов (BIO формат)
        label2id = {"O": 0, "B-ERR": 1, "I-ERR": 2, "B-PRT": 3, "I-PRT": 4, "B-MOD": 5, "I-MOD": 6}
        id2label = {v: k for k, v in label2id.items()}

        def align_labels(examples):
            tokenized_inputs = tokenizer(examples["category"], truncation=True, padding="max_length", max_length=128)
            labels = []
            for i in range(len(examples["category"])):
                # Упрощенная логика разметки для обучения на базе сгенерированных Gemini данных
                word_ids = tokenized_inputs.word_ids(batch_index=i)
                label_ids = [-100 if idx is None else 0 for idx in word_ids]
                labels.append(label_ids)
            tokenized_inputs["labels"] = labels
            return tokenized_inputs

        ds = raw_datasets.map(align_labels, batched=True, remove_columns=raw_datasets["train"].column_names)

        from transformers import AutoModelForTokenClassification
        model = AutoModelForTokenClassification.from_pretrained(
            model_id, num_labels=len(label2id), id2label=id2label, label2id=label2id
        ).to(self.device)

        args = TrainingArguments(
            output_dir=str(config.NER_SAVE_PATH),
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=config.BATCH_SIZE,
            num_train_epochs=3,
            fp16=torch.cuda.is_available(),
            load_best_model_at_end=True
        )

        trainer = Trainer(
            model=model, args=args, train_dataset=ds["train"], eval_dataset=ds["test"],
            data_collator=DataCollatorForTokenClassification(tokenizer)
        )
        trainer.train()
        model.save_pretrained(config.NER_SAVE_PATH)
        tokenizer.save_pretrained(config.NER_SAVE_PATH)
        print(f"NER model saved to {config.NER_SAVE_PATH}")


if __name__ == "__main__":
    trainer = ApplianceTrainer()
    trainer.train_classifier()
    trainer.train_generator()