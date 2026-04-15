import argparse
import sys
from src.labeler import DataLabeler
from src.trainer import ApplianceTrainer
from src.pipeline import ApplianceRepairPipeline
import config


def main():
    parser = argparse.ArgumentParser(description="Appliance Repair NLP Pipeline")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["labeling", "train", "run"],
        required=True,
        help="Mode to run: 'labeling' (Gemini API), 'train' (Local ML), or 'run' (Inference)"
    )
    parser.add_argument("--input", type=str, help="Path to input Excel file (for 'run' mode)")
    parser.add_argument("--output", type=str, help="Path to output Excel file (for 'run' mode)")

    args = parser.parse_args()

    if args.mode == "labeling":
        print("--- Starting Gemini Labeling Process ---")
        labeler = DataLabeler()
        labeler.run()

    elif args.mode == "train":
        print("--- Starting Model Training ---")
        trainer = ApplianceTrainer()
        trainer.train_classifier()
        trainer.train_generator()
        trainer.train_ner()
        print("Training complete. Models saved to /models directory.")

    elif args.mode == "run":
        if not config.CLASSIFIER_SAVE_PATH.exists() or not config.GENERATOR_SAVE_PATH.exists():
            print("Error: Models not found. Run --mode train first.")
            sys.exit(1)

        input_path = args.input if args.input else config.INPUT_EXCEL
        output_path = args.output if args.output else config.DATA_DIR / "processed" / "final_results.xlsx"

        print(f"--- Running Inference Pipeline ---")
        pipeline = ApplianceRepairPipeline()
        pipeline.process_file(input_path, output_path)


if __name__ == "__main__":
    main()