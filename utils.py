import requests

def try_get_request(kwargs):
    try:
        response = requests.get(**kwargs)
        response.raise_for_status()
        return response
    except requests.exceptions.RequestException as e:
        print(e)