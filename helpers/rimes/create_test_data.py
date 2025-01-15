import json
import pandas as pd

def main():
    folders = ['bluche', 'flor', 'puigcerver']

    for folder in folders:
        evaluations_path = f'rimes-results/{folder}/artifacts/evaluations.json'

        with open(evaluations_path, 'r') as f:
            lines = json.loads(f.read())

            predictions = []
            ground_truths = []

            for line in lines:
                prediction = line['predictions'][0]
                predictions.append(prediction['text'])
                ground_truths.append(line['text'])

            df = pd.DataFrame({'prediction': predictions, 'ground_truth': ground_truths})

            df.to_csv(f'datasets/rimes/test_data/{folder}.csv', index=False)
            
if __name__ == '__main__':
    main()