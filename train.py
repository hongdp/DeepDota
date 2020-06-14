import tensorflow as tf

TRAINING_DATA_FILE = 'data/training_data_50k_train.tfrecord'
EVAL_DATA_FILE = 'data/training_data_50k_test.tfrecord'
MODEL_PATH = 'models/prediction_model_32'


def _get_feature_parser():
    feature_description = {
        'radiant_heros': tf.io.FixedLenFeature([5], tf.int64, default_value=[0]*5),
        'dire_heros': tf.io.FixedLenFeature([5], tf.int64, default_value=[0]*5),
    }
    label_description = {'radiant_win': tf.io.FixedLenFeature(
        [], tf.float32, default_value=0)}

    def _parse_function(example_proto):
        # Parse the input `tf.Example` proto using the dictionary above.
        parsed_features = tf.io.parse_single_example(
            example_proto, feature_description)
        parsed_labels = tf.io.parse_single_example(
            example_proto, label_description)

        return parsed_features, parsed_labels['radiant_win']

    return _parse_function


class MyModel(tf.keras.Model):

    def __init__(self):
        super(MyModel, self).__init__()
        # Embedding layer with size = 32.
        embedding_size = 32
        # kernel_regularizer = tf.keras.regularizers.l2(0.001)
        # bias_regularizer = tf.keras.regularizers.l2(0.001)
        kernel_regularizer = None
        bias_regularizer = None
        self.embed = tf.keras.layers.Embedding(
            150, embedding_size, input_length=5, embeddings_regularizer=kernel_regularizer)
        self.dense_layers = []
        self.dense_layers.append(
            tf.keras.layers.Dense(embedding_size, activation=tf.nn.relu, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer))
        self.dense_layers.append(
            tf.keras.layers.Dense(embedding_size//2, activation=tf.nn.relu, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer))
        self.dense_layers.append(
            tf.keras.layers.Dense(1, activation=tf.nn.sigmoid, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer))

    def call(self, inputs):
        '''
        inputs: {
            'dire_heros': <tf.Tensor: id=66, shape=(5,), dtype=int64, numpy=array([ 2, 72, 25, 87, 11])>,
            'radiant_heros': <tf.Tensor: id=67, shape=(5,), dtype=int64, numpy=array([ 91,  52, 104,  86,  17])>,
            'radiant_win': <tf.Tensor: id=68, shape=(), dtype=float32, numpy=0.0>
        }
        '''
        embed_radiant_val = tf.math.reduce_sum(
            self.embed(inputs['radiant_heros']), axis=1)
        embed_dire_val = tf.math.reduce_sum(
            self.embed(inputs['dire_heros']), axis=1)
        x = tf.concat([embed_radiant_val, embed_dire_val], axis=1)
        for dense_layer in self.dense_layers:
            x = dense_layer(x)
        return x


def main():
    training_dataset = tf.data.TFRecordDataset(TRAINING_DATA_FILE)
    training_dataset = training_dataset.map(_get_feature_parser())
    training_dataset = training_dataset.shuffle(10000).batch(1024)

    eval_dataset = tf.data.TFRecordDataset(EVAL_DATA_FILE)
    eval_dataset = eval_dataset.map(_get_feature_parser())
    eval_dataset = eval_dataset.batch(1024)

    model = MyModel()
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss=tf.keras.losses.BinaryCrossentropy(), metrics=['binary_accuracy'])

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='logs/')
    model.fit(training_dataset, epochs=64, callbacks=[
              tensorboard_callback], validation_data=eval_dataset)
    model.save(MODEL_PATH)


if __name__ == '__main__':
    main()
