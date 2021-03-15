import argparse
import tensorflow as tf 

def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        metavar='C',
        default='None',
        help='The Configuration file')
    args = argparser.parse_args()
    return args


def save_model_checkpoint(model_to_save: tf.Module, model_dir: str):
    tf.train.Checkpoint(model=model_to_save).write(model_dir)

def save_model(model_to_save: tf.Module, model_dir: str):
    tf.saved_model.save(model_to_save, model_dir)

def load_model(model_dir: str):
    tf.saved_model.load(model_dir)