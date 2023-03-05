import pandas as pd


def process_dataset(datasets):
    print('datasets', datasets)
    if datasets.find(';') != -1:
        li = []
        for filename in datasets.split(';'):
            df = pd.read_csv(filename, index_col=None, header=0)
            li.append(df)
        balance_data = pd.concat(li, axis=0, ignore_index=True)
        print('len-balance_data(files)', len(balance_data))
    else:
        balance_data = pd.read_csv(datasets, sep=',', header=0)
        print('len-balance_data(one)', len(balance_data))

    return balance_data
