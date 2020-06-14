import pickle
import glob
import tensorflow as tf
from tqdm import tqdm

DATA_FILE_PATTERN = 'data/match_detail_50k.binary.*'
TRAIN_TF_RECORD_FILE = 'data/training_data_50k_train.tfrecord'
TEST_TF_RECORD_FILE = 'data/training_data_50k_test.tfrecord'
TRAIN_TEST_RATIO = 9


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
    test_to_all_ratio = TRAIN_TEST_RATIO + 1
    counter = 0
    with tf.io.TFRecordWriter(TRAIN_TF_RECORD_FILE) as train_writer:
        with tf.io.TFRecordWriter(TEST_TF_RECORD_FILE) as test_writer:
            for idx, file in enumerate(files):
                print('processing shard %d' % idx)
                with open(file, 'rb') as f:
                    all_match_details = pickle.load(f)

                    for match in tqdm(all_match_details):
                        if match == None:
                            continue
                        example = SerializeExample(match)
                        if example:
                            if counter % test_to_all_ratio == 0:
                                test_writer.write(example)
                            else:
                                train_writer.write(example)
                            counter += 1
                        else:
                            print("invalid match")
                            continue


if __name__ == '__main__':
    main()
