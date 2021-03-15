import tensorflow as tf 

def loss(target_y, predicted_y):
  return tf.reduce_mean(tf.square(target_y - predicted_y))

def accuracy(model, set):
    correct_predictions = 0
    for i, l in set:
        if l == model(i):
            correct_predictions += 1

    return 100 * correct_predictions / len(set)