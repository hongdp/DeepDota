import pickle
import glob
import tensorflow as tf
from tqdm import tqdm

DATA_FILE_PATTERN = 'data/match_detail_50000.binary.*'
TF_RECORD_FILE = 'data/training_data_50k.tfrecord'


def SerializeExample(match):
    radiant_heros = []
    dire_heros = []
    for player in match['players']:
        if player['hero_id'] == None:
            return None
        if player['isRadiant'] == None:
            return None
        elif player['isRadiant'] == 1:
            radiant_heros.append(player['hero_id'])
        else:
            dire_heros.append(player['hero_id'])
    if not len(radiant_heros) == 5 or not len(dire_heros) == 5 or match['radiant_win'] == None:
        return None
    feature = {
        'radiant_win': tf.train.Feature(float_list=tf.train.FloatList(value=[float(match['radiant_win'])])),
        'radiant_heros': tf.train.Feature(int64_list=tf.train.Int64List(
            value=radiant_heros)),
        'dire_heros': tf.train.Feature(int64_list=tf.train.Int64List(
            value=dire_heros))}
    example_proto = tf.train.Example(
        features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def main():
    files = glob.glob(DATA_FILE_PATTERN)
    with tf.io.TFRecordWriter(TF_RECORD_FILE) as writer:
        for idx, file in enumerate(files):
            print('processing shard %d' % idx)
            with open(file, 'rb') as f:
                all_match_details = pickle.load(f)
                for match in tqdm(all_match_details):
                    if match == None:
                        continue
                    example = SerializeExample(match)
                    if example:
                        writer.write(example)
                    else:
                        print("invalid match")
                        continue


if __name__ == '__main__':
    main()
