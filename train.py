import tensorflow as tf

TRAINING_DATA_FILE = 'training_data.tfrecord'


def _get_feature_parser():
    feature_description = {
        'radiant_win': tf.io.FixedLenFeature([], tf.float32, default_value=0),
        'radiant_heros': tf.io.FixedLenFeature([5], tf.int64, default_value=[0]*5),
        'dire_heros': tf.io.FixedLenFeature([5], tf.int64, default_value=[0]*5),
    }

    def _parse_function(example_proto):
        # Parse the input `tf.Example` proto using the dictionary above.
        return tf.io.parse_single_example(example_proto, feature_description)

    return _parse_function


class MyModel(tf.keras.Model):

    def __init__(self):
        super(MyModel, self).__init__()
        self.dense1 = tf.keras.layers.Embedding(150, 32, )


def main():
    raw_dataset = tf.data.TFRecordDataset(TRAINING_DATA_FILE)
    parsed_dataset = raw_dataset.map(_get_feature_parser())
    for parsed_record in parsed_dataset.take(10):
        print('taking')
        print(repr(parsed_record))


if __name__ == '__main__':
    main()
