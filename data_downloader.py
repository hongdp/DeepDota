import requests
import pickle

from functools import partial
from multiprocessing import Pool
from tqdm import tqdm

NUM_MATCHES = 50000
MATCH_LIST_FILE = 'match_data_50000.binary'
MATCH_DETAILS_FILE = 'match_detail_50000.binary'
DOWNLOAD_MATCH_LIST = False
DOWNLOAD_MATCH_DETAILS = True
MAX_MATCHES_PER_SHARD = 10000
API_KEY_FILE = 'API_KEY'


def download_match_list(num_matches, api_key):
    all_matches = []
    max_match_id = None
    while len(all_matches) < num_matches:
        if max_match_id:
            request_url = 'https://api.opendota.com/api/proMatches?less_than_match_id=%d&api_key=%s' % (
                max_match_id, api_key)
        else:
            request_url = 'https://api.opendota.com/api/proMatches?api_key=%s' % (
                api_key)
        res = requests.get(request_url)
        all_matches.extend(res.json())
        max_match_id = all_matches[-1]['match_id']
        print('loaded %d of %d matches' % (len(all_matches), num_matches))
    return all_matches[:num_matches]


def download_match_detail(match, api_key):
    match_id = match['match_id']
    res = requests.get(
        'https://api.opendota.com/api/matches/%d?api_key=%s' % (match_id, api_key))
    if res.status_code == 200:
        match.update(res.json())
    else:
        match = None
        print('failed: match %d' % (match_id))
    return match


def main():
    with open(API_KEY_FILE, 'r') as f:
        api_key = f.readline()

    if DOWNLOAD_MATCH_LIST:
        match_list = download_match_list(NUM_MATCHES, api_key)
        with open(MATCH_LIST_FILE, 'w+b') as f:
            pickle.dump(match_list, f)
    else:
        with open(MATCH_LIST_FILE, 'rb') as f:
            match_list = pickle.load(f)[:NUM_MATCHES]
    if DOWNLOAD_MATCH_DETAILS:
        download_match_detail_with_api = partial(
            download_match_detail, api_key=api_key)
        num_shard = len(match_list)//MAX_MATCHES_PER_SHARD + 1
        with Pool(4) as pool:
            for shard in range(num_shard):
                print('downloading shard %d' % (shard))
                begin = shard*MAX_MATCHES_PER_SHARD
                end = (shard+1)*MAX_MATCHES_PER_SHARD if (shard+1) * \
                    MAX_MATCHES_PER_SHARD < len(match_list) else len(match_list)
                shard_matches = match_list[begin: end]
                match_details = list(tqdm(pool.imap(
                    download_match_detail_with_api, shard_matches), total=len(shard_matches)))
                with open(MATCH_DETAILS_FILE+'.%d-of-%d' % (shard, num_shard), 'w+b') as f:
                    pickle.dump(match_details, f)


if __name__ == '__main__':
    main()
