import pickle
import glob
import tensorflow as tf

DATA_FILE_PATTERN = 'match_detail_50000.binary.*'
TF_RECORD_FILE = 'training_data.tfrecord'


def serialize_example(match):
    radiant_heros = []
    dire_heros = []
    for player in match['players']:
        if player['isRadiant']:
            radiant_heros.append(player['hero_id'])
        else:
            dire_heros.append(player['hero_id'])
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
        for file in files:
            with open(file, 'rb') as f:
                all_match_details = pickle.load(f)
                for match in all_match_details:
                    example = serialize_example(match)
                    writer.write(example)


if __name__ == '__main__':
    main()
