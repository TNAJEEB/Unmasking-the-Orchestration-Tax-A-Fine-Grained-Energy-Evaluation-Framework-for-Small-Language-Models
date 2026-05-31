import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import csv
import os
import gc
from tqdm import tqdm
from data_loader import get_all_23_datasets
from telemetry import FullSystemMonitor

def run_inference_loop(model, tokenizer, datasets_dict, csv_filenames, error_log_filenames, model_name):
    monitor = FullSystemMonitor(gpu_indices=[0])
    
    # 1. Write Category 1, 3, and 6 Headers
    for csv_filename in csv_filenames:
        write_header = not os.path.exists(csv_filename)
        with open(csv_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            if write_header:
                writer.writerow([
                    "Model", "Dataset", "Sample_ID", "Input_Tokens", "Output_Tokens", 
                    "Prefill_GPU_J", "Prefill_CPU_J", "Prefill_RAM_J", "Prefill_Time_s",
                    "Decode_GPU_J", "Decode_CPU_J", "Decode_RAM_J", "Decode_Time_s"
                ])

    # 2. Write Error Log Headers
    for err_filename in error_log_filenames:
        write_err_header = not os.path.exists(err_filename)
        with open(err_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            if write_err_header:
                writer.writerow(["Model", "Dataset", "Original_Index", "Error_Type", "Detailed_Message"])

    print(f"\n--- Warming up GPU for {model_name} to stabilize thermal leakage ---")
    dummy_input = torch.randint(0, 1000, (1, 512)).to('cuda')
    with torch.no_grad():
        for _ in range(50):
            _ = model(dummy_input)
    torch.cuda.synchronize()
    print("Warm-up complete. Starting benchmark loops...")

    for dataset_name, samples in datasets_dict.items():
        print(f"\n--- Running {model_name} | Dataset: {dataset_name} ---")
        
        successful_samples = 0
        target_samples = 1000
        
        # Manually control tqdm so it tracks successful saves, not total attempts
        pbar = tqdm(total=target_samples, desc=f"Evaluating {dataset_name}")
        
        for i, prompt in enumerate(samples):
            # Stop if we have successfully collected exactly 1000 valid samples
            if successful_samples >= target_samples:
                break
                
            try:
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
                input_length = inputs.input_ids.shape[1]

                # ==========================================
                # VOCABULARY SAFETY SHIELD
                # ==========================================
                if hasattr(model.config, 'vocab_size') and inputs.input_ids.max() >= model.config.vocab_size:
                    error_msg = f"Token ID ({inputs.input_ids.max().item()}) exceeds vocab size ({model.config.vocab_size})"
                    for err_filename in error_log_filenames:
                        with open(err_filename, mode='a', newline='') as f:
                            csv.writer(f).writerow([model_name, dataset_name, i, "Vocab_Assert", error_msg])
                    continue # Skip to next sample WITHOUT incrementing successful_samples
                # ==========================================

                # PREFILL PHASE
                monitor.start()
                t_prefill_start = time.time()
                with torch.no_grad():
                    outputs = model(**inputs, use_cache=True)
                    next_token_logits = outputs.logits[:, -1, :]
                    next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
                    past_key_values = outputs.past_key_values
                
                torch.cuda.synchronize()
                t_prefill = time.time() - t_prefill_start
                prefill_data = monitor.stop()

                # DECODE PHASE
                monitor.start()
                t_decode_start = time.time()
                max_new_tokens = 50 
                tokens_generated = 1
                
                with torch.no_grad():
                    for _ in range(max_new_tokens - 1):
                        outputs = model(input_ids=next_token, past_key_values=past_key_values, use_cache=True)
                        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(0)
                        past_key_values = outputs.past_key_values
                        tokens_generated += 1
                        if next_token.item() == tokenizer.eos_token_id:
                            break
                
                torch.cuda.synchronize()
                t_decode = time.time() - t_decode_start
                decode_data = monitor.stop()

                # SAVE TO ALL LOCATIONS
                row = [
                    model_name, dataset_name, i, input_length, tokens_generated,
                    round(prefill_data["Total_GPU_Joules"], 4), round(prefill_data["CPU_Joules"], 4), round(prefill_data["RAM_Joules"], 4), round(t_prefill, 4),
                    round(decode_data["Total_GPU_Joules"], 4), round(decode_data["CPU_Joules"], 4), round(decode_data["RAM_Joules"], 4), round(t_decode, 4)
                ]

                for csv_filename in csv_filenames:
                    with open(csv_filename, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(row)

                # Memory cleanup
                del inputs, outputs, next_token_logits, next_token, past_key_values
                if i % 100 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()

                # Increment success tracker and update progress bar
                successful_samples += 1
                pbar.update(1)

            except Exception as e:
                # Catch ANY other error (OOM, missing keys, etc.), log it, and pull a new sample
                for err_filename in error_log_filenames:
                    with open(err_filename, mode='a', newline='') as f:
                        csv.writer(f).writerow([model_name, dataset_name, i, "General_Exception", str(e)])
                continue

        pbar.close()
        
        if successful_samples < target_samples:
            print(f"\nWARNING: Dataset {dataset_name} exhausted. Only secured {successful_samples}/{target_samples} samples.")

if __name__ == "__main__":
    MODEL_LIST = [
        # --- COMPLETED MODELS (COMMENTED OUT) ---
        
        # --- PROBLAMATIC MODELS ---
        # "EleutherAI/gpt-neo-1.3B",
        # "databricks/dolly-v2-3b",
        
        # --- PENDING MODELS ---
        "Qwen/Qwen2.5-1.5B",
        "microsoft/phi-1_5",
        "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
        "EleutherAI/pythia-2.8b",
        "microsoft/Phi-3-mini-4k-instruct",
        "meta-llama/Llama-3.2-1B",
        "google/gemma-2b",
        "princeton-nlp/Sheared-LLaMA-2.7B",
        "openlm-research/open_llama_3b_v2",
        "stabilityai/stablelm-3b-4e1t",
        "google/gemma-3-1b-it",
        "google/gemma-4-E2B-it",
        "mistralai/Mistral-7B-v0.1",
        "meta-llama/Llama-2-7b-hf",
        "HuggingFaceH4/zephyr-7b-beta"
    ] 

    OUTPUT_DIRS = [
        "/home/talha/thesis/SLM Bench/Implementation_RTX5060/Phi-1.5B", 
        "/home/talha/thesis_data"                                       
    ]

    for d in OUTPUT_DIRS:
        os.makedirs(d, exist_ok=True)
    
    all_datasets = get_all_23_datasets()
    
    for MODEL_NAME in MODEL_LIST:
        print(f"\n>>> INITIATING THESIS BENCHMARK: {MODEL_NAME}")
        
        monitor = FullSystemMonitor(gpu_indices=[0])
        print("\nMeasuring Category 2: Setup Stages...")
        
        monitor.start()
        t0 = time.time()
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        torch.cuda.synchronize()
        t_tokenizer = time.time() - t0
        tok_energy = monitor.stop()
        
        monitor.start()
        t1 = time.time()
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="cuda")
        torch.cuda.synchronize()
        t_model = time.time() - t1
        mod_energy = monitor.stop()
        
        clean_name = MODEL_NAME.replace('/','_')
        for d in OUTPUT_DIRS:
            cat2_file = os.path.join(d, f"{clean_name}_Cat2_Lifecycle.csv")
            with open(cat2_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Model", "Stage", "Time_s", "GPU_J", "CPU_J", "RAM_J"])
                writer.writerow([MODEL_NAME, "Tokenizer_Init", round(t_tokenizer, 4), round(tok_energy["Total_GPU_Joules"], 4), round(tok_energy["CPU_Joules"], 4), round(tok_energy["RAM_Joules"], 4)])
                writer.writerow([MODEL_NAME, "Model_Init", round(t_model, 4), round(mod_energy["Total_GPU_Joules"], 4), round(mod_energy["CPU_Joules"], 4), round(mod_energy["RAM_Joules"], 4)])

        print("\nStarting Main Inference Run (Categories 1, 3, and 6)...")
        
        # Pass both the main data files AND the error log files into the loop
        main_csv_paths = [os.path.join(d, f"{clean_name}_Cat1_3_6_Inference.csv") for d in OUTPUT_DIRS]
        error_log_paths = [os.path.join(d, f"{clean_name}_Error_Log.csv") for d in OUTPUT_DIRS]
        
        run_inference_loop(model, tokenizer, all_datasets, main_csv_paths, error_log_paths, MODEL_NAME)
        
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
