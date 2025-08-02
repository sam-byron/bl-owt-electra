from transformers import AutoModel, AutoTokenizer
import os

def main():
    # model_name = "google-bert/bert-base-uncased"
    model_name = "ltg/gpt-bert-babylm-base"
    # output_dir = "bert-base-uncased-saved"
    output_dir = "gpt-bert-babylm-base-saved"
    os.makedirs(output_dir, exist_ok=True)

    # Load model & tokenizer
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Save both to the same directory
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"✅ Saved model and tokenizer to: {output_dir}")

if __name__ == "__main__":
    main()