import pandas as pd

def main():
    base_url = 'https://raw.githubusercontent.com/savi8sant8s/ptbr-post-ocr-sc-llm/refs/heads/main'

    models = ['azure', 'bluche', 'flor', 'puigcerver', 'ltu_main', 'ltu_ensemble', 'pero', 'demokritos', 'litis']

    df_ground_truth = pd.read_csv(f'{base_url}/test_ground_truth.csv')

    for model in models:
        df_model = pd.read_csv(f'{base_url}/test_data/{model}.csv')
        df_model['ground_truth'] = df_ground_truth['ground_truth']
        df_model.to_csv(f'datasets/bressay/test_data/{model}.csv', index=False)
        
if __name__ == '__main__':
    main()