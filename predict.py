import tensorflow as tf
import numpy as np

MODEL_PATH = 'models/prediction_model_16'


def GetHerosFromStdIn(list_name):
    '''
    Args:
        list_name: A string used in the prompt to indicate which team the heros_list is for.
    Return:
        A np array of 5 hero ids.
    '''
    heros_list = []
    while not len(heros_list) == 5:
        heros_list.clear()
        heros_str = input(
            'Please provide 5 heros for %s. Comma separated:' % (list_name))
        heros_str_list = heros_str.split(',')
        for hero in heros_str_list:
            try:
                heros_list.append(int(hero))
            except:
                continue

    return np.array(heros_list).reshape(1, 5)


def main():
    model = tf.keras.models.load_model(MODEL_PATH)
    while True:
        radiant_heros = GetHerosFromStdIn('radiant')
        dire_heros = GetHerosFromStdIn('dire')
        print(radiant_heros)
        print(dire_heros)
        radiant_win_rate = model.predict({'radiant_heros': radiant_heros,
                                          'dire_heros': dire_heros})
        print('Predicted radiant winning rate: %f' % (radiant_win_rate))


if __name__ == '__main__':
    main()
