from collections import defaultdict, OrderedDict
from itertools import chain, product
import json
import numpy as np
from sklearn.model_selection import train_test_split
import utils


class SourceData:
# https://www.disprot.org/help#api

    @staticmethod
    def get_ids():
        response = utils.try_get_request({'url': 'https://www.disprot.org/api/list_ids'})
        ids = json.loads(response.text)
        return ids['disprot_ids']

    @staticmethod
    def get_data(id):
        response = utils.try_get_request({
            'url': 'https://www.disprot.org/api/' + id,
            'params': OrderedDict([
                ('show_ambiguous', 'true'),
                ('show_obsolete', 'true'),
                ('release', '2019_09'),
                ('format', 'json')
            ])
        })
        data = json.loads(response.text)
        X = np.array([ord(amino_acid) for amino_acid in data['sequence']])
        Y = SourceData._get_sequence_labels(data['length'],
                                            data['disprot_consensus']['full'])
        return X, Y

    @staticmethod
    def _get_sequence_labels(length, consensus):
        Y = np.full(length, np.nan)
        for region in consensus:
            Y[region['start']-1:region['end']] = ord(region['type'])
        return Y

    @staticmethod
    def load_data_sets(file_arg='data.npz'):
        try:
            # Only load data from trusted sources, see warning in documentation:
            # https://numpy.org/doc/stable/reference/generated/numpy.load.html#numpy.load
            data = np.load(file_arg, allow_pickle=True)
            data_set_names = [data_type + '_' + data_set
                              for data_type in ('X', 'Y')
                              for data_set in ('train', 'test')]
            data_sets = list(data[data_set_name]
                              for data_set_name in data_set_names)
        except:
            data_sets = SourceData.get_data_sets(file_arg)
        return data_sets

    @staticmethod
    def get_data_sets(file_arg, get_ids=get_ids, get_data=get_data):
        ids = SourceData.get_ids()
        train_ids, test_ids = train_test_split(ids, test_size=0.2, random_state=42)
        data = defaultdict(list)
        for data_set, set_id in chain(product(['train'], train_ids),
                                    product(['test'], test_ids)):
            X, Y = SourceData.get_data(set_id)
            data['X_' + data_set].append(X)
            data['Y_' + data_set].append(Y)
        data_set_names = [data_type + '_' + data_set
                        for data_type in ('X', 'Y')
                        for data_set in ('train', 'test')]
        return list(data[data_set_name] for data_set_name in data_set_names)
    
    @staticmethod
    def save_data_sets(X_train, X_test, Y_train, Y_test, file_args='data.npz',
                       **kwargs):
        if type(file_args) == str:
            with open(file_args, 'wb') as outfile:
                np.savez_compressed(outfile, X_train=X_train, X_test=X_test,
                         Y_train=Y_train, Y_test=Y_test)
        else:
            try:
                np.savez_compressed(file_args, X_train=X_train, X_test=X_test,
                         Y_train=Y_train, Y_test=Y_test)
            except TypeError as e:
                print('TypeError with file_args in SourceData.save_data_sets()',
                      e)
