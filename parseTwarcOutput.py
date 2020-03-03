import pandas as pd
import numpy as np

def parse(file):
    parsed = []
    with open(file + '_hydrated.txt') as infile:
        for line in infile:
            ID = line.find('"id": ') + 6
            ID_end = line.find(',', ID)
            text = line.find('"full_text": "') + 14
            text_end = line.find('", "', text)
            parsed.append([line[ID:ID_end], line[text:text_end]])
    data = pd.DataFrame(np.array(parsed), columns=['tweet_id', 'content'])
    data.to_csv(file+'_parsed.csv', header=True)
    

if __name__ == '__main__':
    parse('./data/harvard-dataset/test')