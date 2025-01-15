import pandas as pd
import argparse
from src.metrics import print_metrics

def main(dataset, output_corrections, ocr_predictions):
    print("dataset, ocr_prediction, CER_base, WER_base, CER_corr, WER_corr")
    for ocr_prediction in ocr_predictions:
        df_corrections = pd.read_csv(f'{output_corrections}/{ocr_prediction}.csv')
        df_predictions = pd.read_csv(f'datasets/{dataset}/test_data/{ocr_prediction}.csv')
        
        corrections = df_corrections['correction'].tolist()
        inputs = df_predictions['prediction'].tolist()[:len(corrections)]
        outputs = df_predictions['ground_truth'].tolist()[:len(corrections)]
        
        if ocr_prediction == "azure":
            filtered_data = [(inp, out, corr) for inp, out, corr in zip(inputs, outputs, corrections) if type(inp) is str and inp != "SKIP"]
            
            if filtered_data:
                inputs, outputs, corrections = zip(*filtered_data)
            else:
                inputs, outputs, corrections = [], [], []
        
        print_metrics(inputs, outputs, corrections, dataset, ocr_prediction)
    
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', type=str, default='rimes')
  parser.add_argument('--output_corrections', type=str, default='corrections/rimes')
  parser.add_argument('--ocr_predictions', nargs='+', type=str, default=['bluche', 'flor', 'puigcerver'])
  
  args = parser.parse_args()
  
  main(args.dataset, args.output_corrections, args.ocr_predictions)