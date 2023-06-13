import json

import pandas as pd

from sklearn.model_selection import train_test_split


def main():
    train_path = '../data/D80M_train.tsv'
    test_path = '../data/D80M_test.tsv'
    mappings = {'AdvertiserId': json.load(open('../data/advertiser_vocab.json', 'r')),
                'AdKeyword_tokens': json.load(open('../data/keyword_vocab.json', 'r')),
                'AdTitle_tokens': json.load(open('../data/title_vocab.json', 'r')),
                'AdDescription_tokens': json.load(open('../data/description_vocab.json', 'r')),
                'Query_tokens': json.load(open('../data/query_vocab.json', 'r')),
                'AdId': json.load(open('../data/ad_vocab.json', 'r')),
                'DisplayURL': json.load(open('../data/display_vocab.json', 'r')),
                }
    for i, chunk in enumerate(pd.read_csv('../data/D80M.tsv', sep='\t', chunksize=1e7)):
        print(f'{i} / {int(8e7 // 1e7)}', end='\r')
        train_chunk, test_chunk = train_test_split(chunk, test_size=0.1, random_state=42)
        train_chunk = train_chunk.copy()
        test_chunk = test_chunk.copy()
        train_chunk[['AdvertiserId', 'AdId', 'DisplayURL']] = train_chunk[['AdvertiserId', 'AdId', 'DisplayURL']].astype(str)
        test_chunk[['AdvertiserId', 'AdId', 'DisplayURL']] = test_chunk[['AdvertiserId', 'AdId', 'DisplayURL']].astype(str)
        for col, mapping in mappings.items():
            train_chunk[col] = train_chunk[col].apply(lambda x: '|'.join([str(mapping[str(el)]) for el in x.split('|') if str(el) in mapping]))
            test_chunk[col] = test_chunk[col].apply(lambda x: '|'.join([str(mapping[str(el)]) for el in x.split('|') if str(el) in mapping]))
            train_chunk[col] = train_chunk[col].apply(lambda x: x if x else '0')
            test_chunk[col] = test_chunk[col].apply(lambda x: x if x else '0')
        if i == 0:
            train_chunk.to_csv(train_path, sep='\t', index=False)
            test_chunk.to_csv(test_path, sep='\t', index=False)
        else:
            train_chunk.to_csv(train_path, sep='\t', index=False, header=False, mode='a')
            test_chunk.to_csv(test_path, sep='\t', index=False, header=False, mode='a')

    test_df = pd.read_csv('../data/D5M_test_x.tsv.gz', sep='\t', compression='gzip')
    test_df = test_df.copy()
    test_df[['AdvertiserId', 'AdId', 'DisplayURL']] = test_df[['AdvertiserId', 'AdId', 'DisplayURL']].astype(str)
    for col, mapping in mappings.items():
        test_df[col] = test_df[col].apply(lambda x: '|'.join([str(mapping[str(el)]) for el in x.split('|') if str(el) in mapping]))
        test_df[col] = test_df[col].apply(lambda x: x if x else '0')

    test_df.to_csv('../data/D5M_test_x.tsv', sep='\t', index=False)



if __name__ == '__main__':
    main()
