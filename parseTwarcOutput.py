import pandas as pd
import numpy as np

def parse():
    parsed = []
    idx = 4
    size = 1000000
    count = 0
    with open('./data/harvard-dataset/election-filter1_hydrated.txt') as infile:
        for line in infile:
            ID = line.find('"id": ') + 6
            ID_end = line.find(',', ID)
            text = line.find('"full_text": "') + 14
            text_end = line.find('", "', text)
            parsed.append([line[ID:ID_end], line[text:text_end]])
            count += 1
            if count == size/2:
                print("................")
            if count == size:
                target = pd.DataFrame(np.array(parsed), columns=['tweet_id', 'content'])
                target = target.dropna().astype({'tweet_id': 'int64', 'content': str})
                target = target.set_index('tweet_id')
                target = target[~target['content'].str.startswith('RT')]
                target['content'] = target['content'].apply(lambda s: s.encode())
                target['troll'] = False
                print(target.head())
                target.to_csv('./data_parsed/tweet_data_'+str(idx)+'.csv')
                idx += 1
                parsed = []
                count = 0
    target = pd.DataFrame(np.array(parsed), columns=['tweet_id', 'content'])
    target = target.dropna().astype({'tweet_id': 'int64', 'content': str})
    target = target.set_index('tweet_id')
    target = target[~target['content'].str.startswith('RT')]
    target['content'] = target['content'].apply(lambda s: s.encode())
    target['troll'] = False

    target.to_csv('./data_parsed/tweet_data_'+str(idx)+'.csv')

if __name__ == '__main__':
    parse()