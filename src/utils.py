import datetime
import os

import tensorflow as tf


def save(model):
	model_dir = "./models/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
	os.makedirs(model_dir, exist_ok=True)

	# モデル保存
	with open(os.path.join(model_dir, 'model.json'), 'w') as f:
		f.write(model.to_json())

	# モデル重み行列保存
	model.save_weights(os.path.join(model_dir, 'model_weights.hdf5'))

def load(model_dir):
	# モデル読み込み
	with open(os.path.join(model_dir, 'model.json'), 'r') as f:
		model = tf.keras.models.model_from_json(f.read())

	# モデル重み行列読み込み
	model.load_weights(os.path.join(model_dir, 'model_weights.hdf5'))
	return model
