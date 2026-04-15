import pandas as pd
import json
import time
from typing import List
from pydantic import BaseModel, Field
from google import genai
import config


class Entities(BaseModel):
    error_codes: List[str] = Field(description="Error or fault codes")
    part_numbers: List[str] = Field(description="Manufacturer part numbers")
    model_numbers: List[str] = Field(description="Appliance model numbers")


class RepairRecord(BaseModel):
    is_repair_related: bool = Field(description="Is this related to appliance repair?")
    category: str = Field(description="Short descriptive category of the problem")
    has_solution: bool = Field(description="Does the answer contain repair steps?")
    entities: Entities


class DataLabeler:
    def __init__(self):
        self.client = genai.Client(api_key=config.GEMINI_API_KEY)
        self.input_path = config.INPUT_EXCEL
        self.output_path = config.LABELED_DATA
        config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    def _get_processed_ids(self) -> set:
        if not self.output_path.exists():
            return set()
        with open(self.output_path, 'r', encoding='utf-8') as f:
            return {json.loads(line)['original_row_index'] for line in f}

    def run(self, limit: int = config.LABELING_LIMIT):
        if not self.input_path.exists():
            print(f"ERROR: Input file not found at {self.input_path}")
            return

        print(f"Reading {self.input_path}...")
        df = pd.read_excel(self.input_path).head(limit)
        processed_ids = self._get_processed_ids()

        print(f"Starting labeling. Total rows in DF: {len(df)}, Already processed: {len(processed_ids)}")

        if len(df) == 0:
            print("ERROR: Excel file is empty!")
            return

        for index, row in df.iterrows():
            if index in processed_ids:
                continue

            prompt = f"QUESTION: {row.get('Question')}\nANSWER: {row.get('Answer')}"

            while True:
                try:
                    response = self.client.models.generate_content(
                        model=config.GEMINI_MODEL_NAME,
                        contents=prompt,
                        config={
                            "response_mime_type": "application/json",
                            "response_json_schema": RepairRecord.model_json_schema(),
                            "system_instruction": "Label appliance repair data for machine learning training."
                        },
                    )

                    result_json = json.loads(response.text)
                    result_json['original_row_index'] = index

                    with open(self.output_path, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(result_json, ensure_ascii=False) + '\n')

                    print(f"[{index}] Labeled: {result_json['category']}")
                    break

                except Exception as e:
                    print(f"Error at index {index}: {e}. Retrying...")
                    time.sleep(5)


if __name__ == "__main__":
    labeler = DataLabeler()
    labeler.run()