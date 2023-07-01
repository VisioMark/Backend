import numpy as np
import tensorflow as tf

def index_to_label(index: int): 
    shading= {
                    0: 'A',
                    1: 'B',
                    2: 'C', 
                    3: 'D', 
                    4: 'E',
                    5: 'Exception'
                }
    # index = np.apply_along_axis(tf.argmax, 1, predictions)[0]
    return shading.get(index, f"wrong index {index}")


def make_predictions(arr: np.ndarray):
    arr = np.apply_along_axis(lambda x: x/255, 1, arr)
    # arr = np.apply_along_axis(lambda x: np.expand_dims(x, axis=0), 1, arr)
    # load the model
    model = tf.keras.models.load_model("saved_models/model_1.h5")
    # predictions = model.predict(np.expand_dims(arr/255, axis=0))
    predictions = model.predict(arr)
    labels = process_predictions(predictions)
    return labels

def process_predictions(predictions):
    # print(predictions)
    indices = np.apply_along_axis(tf.argmax, 1, predictions)
    labels = np.array(list(map(index_to_label, indices)))
    return labels
