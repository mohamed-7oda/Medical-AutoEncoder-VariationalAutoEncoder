import tensorflow as tf
import os

IMG_SIZE = 64
BATCH_SIZE = 32

def preprocess_image(file_path):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_image(image, channels=1)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32) / 255.0
    return image

def load_dataset(data_dir):
    file_list = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    dataset = tf.data.Dataset.from_tensor_slices(file_list)
    dataset = dataset.map(preprocess_image)
    dataset = dataset.map(lambda x: (x, x))
    dataset = dataset.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return dataset
