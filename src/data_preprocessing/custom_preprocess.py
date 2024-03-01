import pandas as pd
import numpy as np
import spacy

import torch
from torch.nn.utils.rnn import pad_sequence


def custom_preprocess(df, dev_size: float = 0.15, test_size: float = 0.15, check_nan: bool = True, even_dist: bool = True):

    if not ((0.01 <= dev_size <= 0.25) and (0.01 <= test_size <= 0.25)):
        raise ValueError(
            f'Wrong dev and/or test sizes. It should be 2 float numbers (each from 0.01 to 0.25), but received dev_size={dev_size} and test_size={test_size}')

    if even_dist:
        print('Processing started!')
        df = df.sort_values(by='overall')

        texts = np.array(df['reviewText'])
        labels = np.array(df['overall']).reshape(-1, 1)

        if check_nan:
            print('Checking if there are nan values...')
            for i, text in enumerate(texts):
                if type(text) != str:
                    raise OverflowError(
                        f'type of text "{text}" (of index {i}) is not str. You can use df.dropna()')

        print('Converting texts into matrices... (It may take a while, depending on df size)')

        nlp = spacy.load('en_core_web_lg')

        def prepare_text(text):
            doc = nlp(text)
            vectors = torch.tensor(
                np.array([token.vector for token in doc]))
            return vectors

        texts_matrices = [prepare_text(text) for text in texts]

        X = pad_sequence(texts_matrices, batch_first=True)
        y = torch.tensor(labels)

        X = X.movedim(1, 2)
        
        
        half = (len(df) // 2)
        X0, X1 = X[:half], X[half:]
        y0, y1 = y[:half], y[half:]

        print('Dividing dataset into train, dev and test subsets...')
        bord1 = int(len(X) * (1 - (dev_size + test_size))) // 2
        bord2 = int(len(X) * (1 - test_size)) // 2


        X_train = torch.concat([X0[:bord1], X1[:bord1]])
        y_train = torch.concat([y0[:bord1], y1[:bord1]])
        perm = torch.randperm(len(X_train))
        X_train = X_train[perm]
        y_train = y_train[perm]


        X_dev = torch.concat([X0[bord1:bord2], X1[bord1:bord2]])
        y_dev = torch.concat([y0[bord1:bord2], y1[bord1:bord2]])
        perm = torch.randperm(len(X_dev))
        X_dev = X_dev[perm]
        y_dev = y_dev[perm]


        X_test = torch.concat([X0[bord2:], X1[bord2:]])
        y_test = torch.concat([y0[bord2:], y1[bord2:]])
        perm = torch.randperm(len(X_test))
        X_test = X_test[perm]
        y_test = y_test[perm]


        print('Done! returned: (X_train, X_dev, X_test, y_train, y_dev, y_test). Please, ensure that u received them properly (check variable names)')
        return (X_train, X_dev, X_test, y_train, y_dev, y_test)




    else:
        print('Processing started!')
        texts = np.array(df['reviewText'])
        labels = np.array(df['overall']).reshape(-1, 1)

        if check_nan:
            print('Checking if there are nan values...')
            for i, text in enumerate(texts):
                if type(text) != str:
                    raise OverflowError(
                        f'type of text "{text}" (of index {i}) is not str. You can use df.dropna()')

        print('Converting texts into matrices... (It may take a while)')

        def prepare_text(text):
            doc = nlp(text)
            vectors = torch.tensor(
                np.array([token.vector for token in doc]))
            return vectors

        nlp = spacy.load('en_core_web_lg')

        texts_matrices = [prepare_text(text) for text in texts]

        X = pad_sequence(texts_matrices, batch_first=True)
        y = torch.tensor(labels)

        X = X.movedim(1, 2)

        print('Dividing dataset into train, dev and test subsets...')

        bord1 = int(len(X) * (1 - (dev_size + test_size)))
        bord2 = int(len(X) * (1 - test_size))

        X_train, X_dev, X_test = X[:bord1], X[bord1:bord2], X[bord2:]
        y_train, y_dev, y_test = y[:bord1], y[bord1:bord2], y[bord2:]

        print('Done! returned: (X_train, X_dev, X_test, y_train, y_dev, y_test). Please, ensure that u received them properly (check variable names)')
        return (X_train, X_dev, X_test, y_train, y_dev, y_test)
