import pandas as pd
import gzip
import json
import numpy as np


def sample_to_csv(path_to_json_gz_file_ds, size_by_group=5000, destination='./../../sampled_df.csv'):
    print('Started to unpack .json.gz file')
    full_df = getDF(path_to_json_gz_file_ds)

    print('Sampling...')
    df = full_df[['overall', 'reviewText']]
    df = df.dropna()
    df = (df.loc[df['reviewText'].str.len() < 1500]).loc[df['reviewText'].str.len() > 20]

    group0 = df.loc[df['overall'] == 1].sample(size_by_group)
    group0['overall'] = np.zeros_like(group0['overall'])

    group1 = df.loc[df['overall'] == 5].sample(size_by_group)
    group1['overall'] = np.ones_like(group1['overall'])

    sampled_df = pd.concat([group0, group1]).sample(frac=1)
    sampled_df = sampled_df.dropna()

    sampled_df.to_csv(destination)

    print('Done')


def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield json.loads(l)


def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')


if __name__ == '__main__':
    sample_to_csv('./../../data/Electronics_5.json.gz', size_by_group=35000, destination='./../../sampled_df.csv')

