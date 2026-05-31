# data_loader.py
from datasets import load_dataset, concatenate_datasets
import os

# 2026 Security: Hard-block legacy scripts
os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "0"

def get_all_23_datasets():
    datasets_dict = {}
    SEED = 42
    # EXACT COUNTS from image_9492a1.png
    COUNTS = {
        "BoolQ": 15432, "ARC-Easy": 5876, "ARC-Challenge": 2590, "OpenBookQA": 5957,
        "PIQA": 16113, "Hellaswag": 10421, "WinoGrande": 44321, "CommonsenseQA": 12102,
        "GSM8k": 8034, "AQuA": 99765, "RACE-Middle": 24798, "RACE-High": 26982,
        "CoQA": 127542, "e2e_nlg": 50321, "viggo": 9842, "glue_qnli": 104543,
        "bc5cdr": 20764, "conllpp": 23499, "customer_support": 14872, "legal": 49756,
        "reuters": 9623, "covid": 19874, "drop": 96567
    }

    def load_exact(path, target_count, config="default"):
        """Loads data via Parquet; returns None if inaccessible to avoid crashing."""
        base_url = f"hf://datasets/{path}@refs/convert/parquet"
        all_ds = []
        try:
            # Attempt to gather from all standard splits
            for split in ["train", "validation", "test"]:
                try:
                    ds = load_dataset("parquet", data_files=f"{base_url}/{config}/{split}/*.parquet", split="train")
                    all_ds.append(ds)
                except: continue
            
            if not all_ds:
                # Fallback to default config if specific naming fails
                ds = load_dataset("parquet", data_files=f"{base_url}/default/train/*.parquet", split="train")
                all_ds = [ds]

            combined = concatenate_datasets(all_ds)
            return combined.shuffle(seed=SEED).select(range(min(target_count, len(combined))))
        except Exception as e:
            print(f"!!! Skipping {path}: {str(e)[:100]}")
            return None

    print("--- SLM Bench: Executing Resilient Load (Skipping problematic sets) ---")

    # Mapping logic for all 23 domains in image_9567bc.png
    # --- 1. Question Answering ---
    d = load_exact("google/boolq", COUNTS["BoolQ"]); 
    if d: datasets_dict["BoolQ"] = [f"P: {r['passage']}\nQ: {r['question']}\nA:" for r in d]
    
    d = load_exact("ai2_arc", COUNTS["ARC-Easy"], "ARC-Easy"); 
    if d: datasets_dict["ARC-Easy"] = [f"Q: {r['question']}\nChoices: {r['choices']['text']}\nA:" for r in d]
    
    d = load_exact("ai2_arc", COUNTS["ARC-Challenge"], "ARC-Challenge"); 
    if d: datasets_dict["ARC-Challenge"] = [f"Q: {r['question']}\nChoices: {r['choices']['text']}\nA:" for r in d]
    
    d = load_exact("openbookqa", COUNTS["OpenBookQA"], "main"); 
    if d: datasets_dict["OpenBookQA"] = [f"Q: {r['question_stem']}\nChoices: {r['choices']['text']}\nA:" for r in d]
    
    d = load_exact("stanfordnlp/coqa", COUNTS["CoQA"]); 
    if d: datasets_dict["CoQA"] = [f"Story: {r['story']}\nQ: {r['questions']['input_text'][0] if isinstance(r['questions'], dict) else r['questions'][0]}\nA:" for r in d]

    # --- 2. Reasoning ---
    d = load_exact("piqa", COUNTS["PIQA"]); 
    if d: datasets_dict["PIQA"] = [f"Goal: {r['goal']}\n1: {r['sol1']}\n2: {r['sol2']}\nA:" for r in d]
    
    d = load_exact("hellaswag", COUNTS["Hellaswag"]); 
    if d: datasets_dict["Hellaswag"] = [f"Ctx: {r['ctx']}\nChoices: {r['endings']}\nA:" for r in d]
    
    d = load_exact("winogrande", COUNTS["WinoGrande"], "winogrande_xl"); 
    if d: datasets_dict["WinoGrande"] = [f"S: {r['sentence']}\n1: {r['option1']}\n2: {r['option2']}\nA:" for r in d]
    
    d = load_exact("commonsense_qa", COUNTS["CommonsenseQA"]); 
    if d: datasets_dict["CommonsenseQA"] = [f"Q: {r['question']}\nChoices: {r['choices']['text']}\nA:" for r in d]
    
    d = load_exact("drop", COUNTS["drop"]); 
    if d: datasets_dict["drop"] = [f"P: {r['passage']}\nQ: {r['question']}\nA:" for r in d]

    # --- 3. Math & Specialized ---
    d = load_exact("gsm8k", COUNTS["GSM8k"], "main"); 
    if d: datasets_dict["GSM8k"] = [f"Q: {r['question']}\nA:" for r in d]
    
    d = load_exact("aqua_rat", COUNTS["AQuA"]); 
    if d: datasets_dict["AQuA"] = [f"Q: {r['question']}\nOptions: {r['options']}\nA:" for r in d]
    
    d = load_exact("race", COUNTS["RACE-Middle"], "middle"); 
    if d: datasets_dict["RACE-Middle"] = [f"Art: {r['article']}\nQ: {r['question']}\nA:" for r in d]
    
    d = load_exact("race", COUNTS["RACE-High"], "high"); 
    if d: datasets_dict["RACE-High"] = [f"Art: {r['article']}\nQ: {r['question']}\nA:" for r in d]

    # --- 4. Linguistics & Recognition ---
    d = load_exact("glue", COUNTS["glue_qnli"], "qnli"); 
    if d: datasets_dict["glue_qnli"] = [f"Q: {r['question']}\nS: {r['sentence']}\nA:" for r in d]
    
    d = load_exact("conllpp", COUNTS["conllpp"]); 
    if d: datasets_dict["conllpp"] = [f"Tokens: {' '.join(r['tokens'])}\nEnt:" for r in d]
    
    d = load_exact("tner/bc5cdr", COUNTS["bc5cdr"]); 
    if d: datasets_dict["bc5cdr"] = [f"Txt: {' '.join(r['tokens'])}\nMed:" for r in d]

    # --- 5. Domain Specific ---
    d = load_exact("lex_glue", COUNTS["legal"], "eurlex"); 
    if d: datasets_dict["legal"] = [f"Law: {r['text'][:500]}\nCat:" for r in d]
    
    d = load_exact("reuters21578", COUNTS["reuters"], "ModApte"); 
    if d: datasets_dict["reuters"] = [f"Txt: {r['text'][:500]}\nTop:" for r in d]
    
    d = load_exact("tweet_eval", COUNTS["covid"], "sentiment"); 
    if d: datasets_dict["covid"] = [f"Tweet: {r['text']}\nSent:" for r in d]
    
    d = load_exact("e2e_nlg", COUNTS["e2e_nlg"]); 
    if d: datasets_dict["e2e_nlg"] = [f"In: {r['meaning_representation']}\nDesc:" for r in d]
    
    d = load_exact("GEM/viggo", COUNTS["viggo"]); 
    if d: datasets_dict["viggo"] = [f"In: {r['target']}\nDesc:" for r in d]
    
    d = load_exact("bitext/Bitext-customer-support-llm-chatbot-training-dataset", COUNTS["customer_support"]); 
    if d: datasets_dict["customer_support"] = [f"User: {r['instruction']}\nIntent:" for r in d]

    print(f"--- Initialization Complete: Running benchmark on {len(datasets_dict)} valid domains ---")
    return datasets_dict
