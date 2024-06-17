import pandas as pd
import json
import torch
from tqdm import tqdm

MAX_TRAIN_DATA = 500000

def data_preprocess(mode, verbose=False):
    assert mode in ['train', 'test']
    behaviors_path = f"./data/{mode}/{mode}_behaviors.tsv"
    news_path = f"./data/{mode}/{mode}_news.tsv"
    embed_path = f"./data/{mode}/{mode}_entity_embedding.vec"
    
    behaviors = pd.read_csv(behaviors_path, sep='\t')
    news = pd.read_csv(news_path, sep='\t')
    with open(embed_path, 'r') as vecfile:  
        embeds = {line.split('\t')[0]: [float(x) for x in line.split('\t')[1:-1]] for line in vecfile.readlines()}
    
    title_embeds = []
    abstract_embeds = []
    
    for title_json_str in news['title_entities']:
        if type(title_json_str) == type(''):
            title_embeds.append(json.dumps([embeds.get(title_entity['WikidataId'], None) for title_entity in json.loads(title_json_str)]))
        else:
            title_embeds.append(json.dumps([]))
    for abstract_json_str in news['abstract_entities']:
        if type(abstract_json_str) == type(''):
            abstract_embeds.append(json.dumps([embeds.get(abstract_entity['WikidataId'], None) for abstract_entity in json.loads(abstract_json_str)]))
        else:
            abstract_embeds.append(json.dumps([]))
    
    news['title_embeds'] = title_embeds
    news['abstract_embeds'] = abstract_embeds
    news.drop(columns=['title_entities', 'abstract_entities', 'title', 'abstract', 'URL'], inplace=True)
    news.to_csv(f'{mode}/{mode}_news_processed.tsv', index=False, sep='\t')
    
    verbose and print(len(behaviors), len(news), len(embeds))
    verbose and print(behaviors)
    verbose and print(news)
    
def load_data(mode):
    assert mode in ['train', 'test']
    behaviors_path = f"./data/{mode}/{mode}_behaviors.tsv"
    news_path = f"./data/{mode}/{mode}_news_processed.tsv"
    behaviors = pd.read_csv(behaviors_path, sep='\t').sample(frac=1, ignore_index=True, random_state=0)
    news = pd.read_csv(news_path, sep='\t')
    news['title_embeds'] = news['title_embeds'].apply(lambda embeds: json.loads(embeds) if type(embeds) == type('') else None)
    news['abstract_embeds'] = news['abstract_embeds'].apply(lambda embeds: json.loads(embeds) if type(embeds) == type('') else None)
    
    news_dict = {}
    
    for _, row in tqdm(news.iterrows(), total=len(news)):
        embeds = []
        embeds.extend(row['title_embeds'])
        embeds.extend(row['abstract_embeds'])
        embeds = [e for e in embeds if e is not None]
        if len(embeds) == 0: continue
        
        embeds = torch.tensor(embeds)
        embeds = torch.mean(embeds, dim=0, keepdim=True)
        news_dict[row['news_id']] = embeds
        
    raw_datas = []
    
    if mode == 'train':
        for _, row in tqdm(behaviors.iterrows(), total=len(behaviors)):
            clicked_news = row['clicked_news']
            impressions = row['impressions']
            
            clicked_news_embed = [news_dict.get(clicked_new, None) for clicked_new in clicked_news.split()]
            clicked_news_embed = [e.tolist() for e in clicked_news_embed if e is not None]
            if len(clicked_news_embed) == 0: continue
            clicked_news_embed = torch.mean(torch.tensor(clicked_news_embed), dim=0, keepdim=True).squeeze(0)
            
            for impression in impressions.split():
                impression = impression.split('-')
                if len(impression) != 2: continue
                impression_new, clicked = impression
                impression_new_embed = news_dict.get(impression_new, None)
                if impression_new_embed is None: continue
                raw_datas.append([json.dumps(torch.cat([clicked_news_embed, impression_new_embed], dim=-1).tolist()), int(clicked)])
                if len(raw_datas) > MAX_TRAIN_DATA:
                    break
            else:
                continue
            break
    else:
        for _, row in tqdm(behaviors.iterrows(), total=len(behaviors)):
            clicked_news = row['clicked_news']
            impressions = row['impressions']
            clicked_news_embed = [news_dict.get(clicked_new, None) for clicked_new in clicked_news.split()]
            clicked_news_embed = [e.tolist() for e in clicked_news_embed if e is not None]
            clicked_news_embed = torch.mean(torch.tensor(clicked_news_embed), dim=0, keepdim=True).squeeze(0) if len(clicked_news_embed) != 0 else torch.zeros((1, 100))
                
            for impression_new in impressions.split():
                impression_new_embed = news_dict.get(impression_new, torch.zeros((1, 100)))
                raw_datas.append([json.dumps(torch.cat([clicked_news_embed, impression_new_embed], dim=-1).tolist())])
    columns = ['input', 'label'] if mode == 'train' else ['input']
    pd.DataFrame(raw_datas, columns=columns).to_csv(f"result_{mode}.csv", index=False, columns=columns)
    
    return raw_datas

    
    
if __name__ == '__main__':
    # data_preprocess('train')    
    # data_preprocess('test')
    train_data = load_data('train')
    test_data = load_data('test')
    print(train_data[0])
    print(test_data[0])