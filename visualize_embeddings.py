import tensorflow as tf
import os
from tensorboard.plugins import projector


MODEL_PATH = 'models/prediction_model'
LOG_DIR = 'visualization/'


def main():
    model = tf.keras.models.load_model(MODEL_PATH)
    # with open(os.path.join(LOG_DIR, 'metadata.tsv'), "w") as f:
    #     for subwords in encoder.subwords:
    #         f.write("{}\n".format(subwords))
    #     # Fill in the rest of the labels with "unknown"
    #     for unknown in range(1, encoder.vocab_size - len(encoder.subwords)):
    #         f.write("unknown #{}\n".format(unknown))
    weights = tf.Variable(model.embed.get_weights()[0][1:])
    # Create a checkpoint from embedding, the filename and key are
    # name of the tensor.
    checkpoint = tf.train.Checkpoint(embedding=weights)
    checkpoint.save(os.path.join(LOG_DIR, "embedding.ckpt"))
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    # The name of the tensor will be suffixed by `/.ATTRIBUTES/VARIABLE_VALUE`
    embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
    # embedding.metadata_path = 'metadata.tsv'
    projector.visualize_embeddings(LOG_DIR, config)


if __name__ == '__main__':
    main()
