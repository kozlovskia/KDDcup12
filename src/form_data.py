import pandas as pd

from sklearn.model_selection import train_test_split


def main():
    train_path = '../data/D80M_train.tsv'
    test_path = '../data/D80M_test.tsv'
    for i, chunk in enumerate(pd.read_csv('../data/D80M.tsv', sep='\t', chunksize=1e7)):
        print(f'{i} / {int(8e7 // 1e7)}', end='\r')
        train_chunk, test_chunk = train_test_split(chunk, test_size=0.1, random_state=42)
        if i == 0:
            train_chunk.to_csv(train_path, sep='\t', index=False)
            test_chunk.to_csv(test_path, sep='\t', index=False)
        else:
            train_chunk.to_csv(train_path, sep='\t', index=False, header=False, mode='a')
            test_chunk.to_csv(test_path, sep='\t', index=False, header=False, mode='a')



if __name__ == '__main__':
    main()
