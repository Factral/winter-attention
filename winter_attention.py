import torch
import tensorflow as tf
import numpy as np
import pathlib

path_to_zip = tf.keras.utils.get_file(
    'spa-eng.zip', origin='http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip',
    extract=True)

path_to_file = pathlib.Path(path_to_zip).parent/'spa-eng/spa.txt'

def load_data(path):
  text = path.read_text(encoding='utf-8')

  lines = text.splitlines()
  pairs = [line.split('\t') for line in lines]

  print(pairs)

  context = np.array([context for target, context in pairs])
  target = np.array([target for target, context in pairs])

  return target, context

target_raw, context_raw = load_data(path_to_file)
print(context_raw[-1])
