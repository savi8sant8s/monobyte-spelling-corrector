import datasets
import noisocr as ncr
import random as rd
import pandas as pd

def main():
    df = datasets.load_dataset('Teklia/RIMES-2011-line')

    texts = df['train']['text'] + df['validation']['text']

    errors_interactions = [(0,1), (1,3), [3, 6]]

    predictions = []
    ground_truths = []

    for i in range(3):
        for text in texts:
            prediction = ncr.simulate_errors(text, interactions=rd.randint(*errors_interactions[i]))
            predictions.append(prediction)
            ground_truths.append(text)

    #Downloaded from https://wortschatz.uni-leipzig.de/en/download/French - Web - 2011 - France - 30k
    sentences = 'sentences.txt'

    with open(sentences, 'r') as f:
        lines = f.readlines()
        for line in lines:
            index, text = line.strip().split('\t')
            texts = ncr.sliding_window(text)
            for text in texts:
                prediction = ncr.simulate_errors(text, interactions=rd.randint(0, 5))
                predictions.append(prediction)
                ground_truths.append(text)
                
    df = pd.DataFrame({'prediction': predictions, 'ground_truth': ground_truths})

    df = df.sample(frac=1).reset_index(drop=True)

    df = df[df['prediction'].str.len() > 5]

    df.to_csv('datasets/rimes/train_data.csv', index=False)
    
if __name__ == '__main__':
    main()