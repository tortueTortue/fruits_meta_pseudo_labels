import os
import pathlib
import tensorflow as tf
import tensorflow_datasets as tfds

dir_path = os.path.dirname(os.path.realpath(__file__))
dataset_path = os.path.join(dir_path, '../Dataset/Training')

fruit_root = tf.keras.utils.get_file(dataset_path, '')
fruit_root = pathlib.Path(fruit_root)

# for item in fruit_root.glob("*"):
#     print(item.name)

list_ds = tf.data.Dataset.list_files(str(fruit_root / '*/*'))


# for f in list_ds.take(5):
#     print(f.numpy())

def process_path(file_path):
    label = tf.strings.split(file_path, os.sep)[-2]
    return tf.io.read_file(file_path), label


labeled_ds = list_ds.map(process_path)


# for image_raw, label_text in labeled_ds.take(10):
#     print(repr(image_raw.numpy()[:100]))
#     print()
#     print(label_text.numpy())


def split_ds(ds):
    dataset_size = len(ds)
    train_size = int(0.8 * dataset_size)
    test_size = int(0.2 * dataset_size)

    train_dataset = labeled_ds.take(train_size)
    test_dataset = labeled_ds.skip(train_size)
    test_dataset = test_dataset.take(test_size)
    return train_dataset, test_dataset

def get_labels(ds):
    label_set = set([])
    for _, label_text in ds:
        label_set.add(str(label_text.numpy()))
    return label_set


ds_train, ds_test = split_ds(labeled_ds)
print(len(ds_train), len(ds_test))
print(len(get_labels(labeled_ds)))

def fruit360_dataset():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dataset_path = os.path.join(dir_path, '../Dataset/Training')
    fruit_root = tf.keras.utils.get_file(dataset_path, '')
    fruit_root = pathlib.Path(fruit_root)

    list_ds = tf.data.Dataset.list_files(str(fruit_root / '*/*'))

    def process_path(file_path):
        label = tf.strings.split(file_path, os.sep)[-2]
        return tf.io.read_file(file_path), label

    labeled_ds = list_ds.map(process_path)
    return split_ds(labeled_ds), get_labels(labeled_ds)
