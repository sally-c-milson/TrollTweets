import pandas as pd

if __name__ == '__main__':
    for i in range(13):
        poss = pd.read_csv('./data_parsed/tweet_data_'+str(i)+'.csv')
        poss = poss.set_index('tweet_id')
        data = [poss]
        for j in range(14):
            idx = 13 + i*14 + j
            neg = pd.read_csv('./data_parsed/tweet_data_'+str(idx)+'.csv')
            neg = neg.set_index('tweet_id')
            data.append(neg)
        totalSet = pd.concat(data)
        totalSet = totalSet.loc[~totalSet.index.duplicated(keep='first')]
        print(len(totalSet))
        totalSet.to_csv('./data/tweet_data_batch'+str(i)+'.csv', header=True)