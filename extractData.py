import pandas as pd

# 3M - https://github.com/fivethirtyeight/russian-troll-tweets/
# After: 1.3M
def filterRetweets():
    data = []
    for i in range(1,14):
        target = pd.read_csv('./data/russian-troll-tweets/IRAhandle_tweets_'+str(i)+'.csv')
        target = target[target['retweet'] == True]
        target = target[['content','tweet_id' ]]
        target = target.set_index('tweet_id')
        target['content'] = target['content'].apply(lambda s: s.encode())
        target['troll'] = True
        data.append(target)
    totalSet = pd.concat(data)
    totalSet = totalSet.loc[~totalSet.index.duplicated(keep='first')]
    print(len(totalSet))
    print(totalSet.head())
    totalSet.to_csv('tweet_data.csv', header=True)

def mergeData(file, cols, headers, encoding):
    if headers:
        target = pd.read_csv(file, encoding=encoding)
        target = target.rename(columns=cols)
    else:
        target = pd.read_csv(file, encoding=encoding, names=cols)

    target = target[['content','tweet_id']]
    target = target.dropna().astype({'tweet_id': 'int64', 'content': str})
    target = target.set_index('tweet_id')
    target = target[~target['content'].str.startswith('RT')]
    target['content'] = target['content'].apply(lambda s: s.encode())
    target['troll'] = False

    print(len(target))
    print(target.head())
    print('.............................')

    data = pd.read_csv('tweet_data.csv', index_col='tweet_id', dtype={'tweet_id': 'int64', 'content': str, 'troll': bool})

    totalSet = pd.concat([data,target], sort=True)
    totalSet = totalSet.loc[~totalSet.index.duplicated(keep='first')]
    print(len(totalSet))
    print(totalSet.head())
    print('.............................')
    totalSet.to_csv('tweet_data.csv', header=True)

# for https://data.world/fivethirtyeight/twitter-ratio
def tweetIDFromURL(file):
    target = pd.read_csv(open(file, 'r'), encoding='latin-1')
    tids = target['url'].str.split('/')
    target['tweet_id'] = tids.str[-1]
    target.dropna().to_csv(file, header=True)



if __name__ == '__main__':
    # It is important this is run first because of the way data is merged
    # will keep first of duplicate ids
    filterRetweets()

    # 4k - https://www.kaggle.com/arathee2/demonetization-in-india-twitter-data
    mergeData('./data/demonetization-tweets.csv', {'text': 'content', 'id': 'tweet_id'}, True, 'latin-1')

    # 6k - https://www.kaggle.com/crowdflower/first-gop-debate-twitter-sentiment
    mergeData('./data/Sentiment.csv', {'text': 'content', 'tweet_id': 'tweet_id'}, True, 'utf-8')

    # 1.6M - https://www.kaggle.com/kazanova/sentiment140
    mergeData('./data/sentiment140/training.1600000.processed.noemoticon.csv', ['target', 'tweet_id', 'date', 'flag', 'author', 'content'], False, 'latin-1')

    # 6k - https://www.kaggle.com/benhamner/clinton-trump-tweets
    mergeData('./data/tweets.csv', {'text': 'content', 'id': 'tweet_id'}, True, 'utf-8')

    # 2k + 3k - https://www.kaggle.com/adhok93/inauguration-and-womensmarch-tweets
    mergeData('./data/inauguration.csv', {'text': 'content', 'id': 'tweet_id'}, True, 'latin-1')
    mergeData('./data/womenmarch.csv', {'text': 'content', 'id': 'tweet_id'}, True, 'latin-1')

    # 395k - https://www.kaggle.com/kinguistics/election-day-tweets#election_day_tweets.csv
    mergeData('./data/election_day_tweets.csv', {'text': 'content', 'id': 'tweet_id'}, True, 'utf-8')

    # 3k + 3k + 24k - https://data.world/fivethirtyeight/twitter-ratio
    tweetIDFromURL('./data/BarackObama.csv')
    mergeData('./data/BarackObama.csv', {'text': 'content', 'tweet_id': 'tweet_id'}, True, 'latin-1')
    tweetIDFromURL('./data/realDonaldTrump.csv')
    mergeData('./data/realDonaldTrump.csv', {'text': 'content', 'tweet_id': 'tweet_id'}, True, 'latin-1')
    tweetIDFromURL('./data/senators.csv')
    mergeData('./data/senators.csv', {'text': 'content', 'tweet_id': 'tweet_id'}, True, 'latin-1')



