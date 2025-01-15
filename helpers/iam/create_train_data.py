import noisocr as ncr
import random as rd
import pandas as pd
import re

def main():
    sets_path = ['trainset.txt', 'validationset1.txt', 'validationset2.txt']

    sets = []

    for path in sets_path:
        with open(f'iam/largeWriterIndependentTextLineRecognitionTask/{path}', 'r') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
            sets.extend(lines)

    lines_path = 'iam/ascii/lines.txt'

    errors_interactions = [(0,1), (1,3), [3, 6]]

    predictions = []
    ground_truths = []

    with open(lines_path, 'r') as f:
        lines = f.readlines()

        for i in range(3):
            for line in lines:
                if line.startswith('#'):
                    continue

                if line.split(' ')[0] in sets:
                    text = ' '.join(line.split(' ')[8:]).replace('|', ' ').strip()
                    prediction = ncr.simulate_errors(text, interactions=rd.randint(*errors_interactions[i]))
                    predictions.append(prediction)
                    ground_truths.append(text)

    # Downloaded from https://wortschatz.uni-leipzig.de/en/download/English - Web - 2002 - United Kingdom - 30k
    sentences = 'sentences.txt'

    def fix_spacing(text):
        return re.sub(r'(\w)([.,!?;:])', r'\1 \2', text)

    with open(sentences, 'r') as f:
        lines = f.readlines()
        for line in lines:
            index, text = line.strip().split('\t')
            text = fix_spacing(text)
            texts = ncr.sliding_window(text)
            for text in texts:
                prediction = ncr.simulate_errors(text, interactions=rd.randint(0, 5))
                predictions.append(prediction)
                ground_truths.append(text)
                
    df = pd.DataFrame({'prediction': predictions, 'ground_truth': ground_truths})

    df = df.sample(frac=1).reset_index(drop=True)

    df = df[df['prediction'].str.len() > 5]

    df.to_csv('datasets/iam/train_data.csv', index=False)
    
if __name__ == '__main__':
    main()