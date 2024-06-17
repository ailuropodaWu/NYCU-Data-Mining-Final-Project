import pandas as pd
import json
import torch
import os

from tqdm import tqdm
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split

def data_parse(df: pd.DataFrame):
    df['input'] = df['input'].apply(lambda input: json.loads(input))
    return df

def train():
    train_data = data_parse(pd.read_csv('./train.csv'))
    # train_data = data_parse(pd.read_csv('./result_train.csv'))
    # embed_dim = torch.tensor(train_data['input'][0]).shape[-1]
    # print(f"Embed Dim: {embed_dim}")
    # for i in tqdm(range(embed_dim)):
    #     train_data[f"input_{i}"] = train_data["input"].apply(lambda x: x[0][i])
    # train_data.to_csv('./train.csv')
    X = train_data.drop(columns=['input', 'label'], axis=1)
    y = train_data['label'].astype(int)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)
    print("Finish data prepare")
    
    model = CatBoostClassifier(iterations=1000, learning_rate=0.1, devices='0')
    model.fit(X_train, y_train, eval_set=(X_val, y_val), plot=True, verbose=True)
    model.save_model('model.pth')

def test():
    model = CatBoostClassifier()
    model.load_model('model.pth')

    test_data = data_parse(pd.read_csv('./test.csv')).drop('input', axis=1)
    print("Finish data prepare")
    # test_data = data_parse(pd.read_csv('./result_test.csv'))
    # # embed_dim = torch.tensor(test_data['input'][0]).shape[-1]
    # # print(f"Embed Dim: {embed_dim}")
    # # for i in tqdm(range(embed_dim)):
    # #     test_data[f"input_{i}"] = test_data["input"].apply(lambda x: x[0][i])
    # # test_data.to_csv('./test.csv')    
    
    predictions = model.predict_proba(test_data)[:, 1]

    num_per_row = 15
    num_rows = len(predictions) // num_per_row
    reshaped_predictions = predictions[:num_rows * num_per_row].reshape((num_rows, num_per_row))

    columns = ['p' + str(i+1) for i in range(num_per_row)]
    results = pd.DataFrame(reshaped_predictions, columns=columns)
    
    filenames = [path[-1] for path in list(os.walk('./out'))]
    v = max([int(filename.replace('prediction', '').replace('.csv', '')) for filename in filenames[0]])
    results.to_csv(f'./out/prediction{v + 1}.csv', index_label='id')

if __name__ == '__main__':
    
    train()
    test()