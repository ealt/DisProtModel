import json
import utils


# https://www.disprot.org/help#api

def get_ids():
    response = utils.try_get_request({'url': 'https://www.disprot.org/api/list_ids'})
    ids = json.loads(response.text)
    return ids['disprot_ids']