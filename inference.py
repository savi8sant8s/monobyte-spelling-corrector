from transformers import T5ForConditionalGeneration, ByT5Tokenizer
import pandas as pd
import torch
import argparse
import os
from tqdm import tqdm
import time

def get_max_length(dataset_name):
    if dataset_name == 'iam':
        return 150
    elif dataset_name == "bressay":
        return 200
    elif dataset_name == "rimes":
        return 150
    else:
        raise ValueError(f"Dataset '{dataset_name}' not supported.")
    
def get_input_prompt(dataset, input):
    if dataset == 'iam':
        prefix = "To correct"
    elif dataset == "bressay":
        prefix = "Corrija"
    elif dataset == "rimes":
        prefix = "Corriger"
    else:
        raise ValueError(f"Dataset '{dataset}' not supported.")
    return f"{prefix}: {input}"

def do_correction(text, model, tokenizer, device, dataset):
    input_text = get_input_prompt(dataset, text)
    max_length = get_max_length(dataset)
    inputs = tokenizer.encode(
        input_text,
        return_tensors='pt',
        max_length=max_length + 10,
        padding='max_length',
        truncation=True
    )
    inputs = inputs.to(device)
    corrected_ids = model.generate(
        inputs,
        max_length=max_length + 10,
        num_beams=5,
        early_stopping=True,
    )
    corrected_sentence = tokenizer.decode(
        corrected_ids[0],
        skip_special_tokens=True
    )
    return corrected_sentence 

def main(model_path, ocr_predictions, dataset, output_folder):
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    tokenizer = ByT5Tokenizer.from_pretrained(model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for ocr_prediction in ocr_predictions:
        start_time = time.time()
        df = pd.read_csv(f'datasets/{dataset}/test_data/{ocr_prediction}.csv')

        os.makedirs(output_folder, exist_ok=True)

        corrections = []
        
        for index, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {ocr_prediction}"):
            corrected_sentence = do_correction(row['prediction'], model, tokenizer, device, dataset)   
            corrections.append(corrected_sentence.strip())
        end_time = time.time()
        total_time = end_time - start_time
        mean_time = total_time / len(df)
        print(f"Processed {len(df)} entries in {total_time:.2f} seconds. Mean time per entry: {mean_time:.4f} seconds.")
        pd.DataFrame({'correction': corrections}).to_csv(f'{output_folder}/{ocr_prediction}.csv', index=False)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='monobyte/byt5-mono-pt-v1')
    parser.add_argument('--dataset', type=str, default='bressay')
    parser.add_argument('--output_folder', type=str, default='corrections/bressay')
    parser.add_argument('--ocr_predictions', nargs='+', type=str, default=['bluche', 'flor', 'puigcerver'])

    args = parser.parse_args()
    
    main(args.model, args.ocr_predictions, args.dataset, args.output_folder)