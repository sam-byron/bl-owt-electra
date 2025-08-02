# finance_chatbot

python3 prepare_data.py --config_path model_babylm_bert.json
python3 prepare_data.py --config_path model_babylm_bert.json --sanitize

python3 data_loader.py --config_path model_babylm_bert.json

python3 transformer_trainer.py --config_path model_babylm_bert.json

lm_eval --model hf-mlm --model_args pretrained=/home/sam-byron/engineering/ML/playground/babylm/bert/model_babylm_bert/checkpoint.pt,backend="mlm" --tasks blimp --device cuda:1 --batch_size 64



# Use `--model hf-mlm` and `--model_args pretrained=$MODEL_PATH,backend="mlm"` if using a custom masked LM.
# Add `--trust_remote_code` if you need to load custom config/model files.

python3 inter_chat_acc.py  --config_path model_open_web_full.json --model_path model_open_web_full/checkpoint.pt

python lora_fine_tuning.py --config_path lora_config.json

**Usage examples:**

1. **With your custom base model + LoRA adapters:**
python3 inter_chat_lora.py --base_model_path ./model_vault/full_owt_run_gpt2_train_ds_only.pt --lora_model_path ./alpaca-lora-owt-gpt2  --config_path model_open_web_full.json

2. **With just LoRA adapters (using standard GPT-2 as base):**
python inter_chat_lora.py \
    --lora_model_path ./alpaca-lora-owt-gpt2

