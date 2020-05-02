from collections import OrderedDict
import json
import numpy as np
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
        X = np.array(list(data['sequence']), dtype=np.unicode_)
        Y = SourceData._get_sequence_labels(data['length'],
                                            data['disprot_consensus']['full'])
        return X, Y

    @staticmethod
    def _get_sequence_labels(length, consensus):
        Y = np.array(['?']*length, dtype=np.unicode_)
        for region in consensus:
            Y[region['start']-1:region['end']] = region['type']
        return Y