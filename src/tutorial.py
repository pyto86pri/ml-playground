import datetime
import os

import tensorflow as tf

from utils import save, load

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

train_idx_0_4 = [y in list(range(5)) for y in y_train]
train_idx_5_9 = [y in list(range(5, 9)) for y in y_train]

test_idx_0_4 = [y in list(range(5)) for y in y_test]
test_idx_5_9 = [y in list(range(5, 9)) for y in y_test]

# 0-4のデータ
x_train_0_4 = x_train[train_idx_0_4]
y_train_0_4 = y_train[train_idx_0_4]
x_test_0_4 = x_test[test_idx_0_4]
y_test_0_4 = y_test[test_idx_0_4]

# 5-9のデータ
x_train_5_9 = x_train[train_idx_5_9]
y_train_5_9 = y_train[train_idx_5_9]
x_test_5_9 = x_test[test_idx_5_9]
y_test_5_9 = y_test[test_idx_5_9]

model = tf.keras.models.Sequential([
	tf.keras.layers.Flatten(input_shape=(28, 28)),
	tf.keras.layers.Dense(100, activation='elu', kernel_initializer='he_normal'),
	tf.keras.layers.BatchNormalization(),
	tf.keras.layers.Dropout(0.2),
	tf.keras.layers.Dense(100, activation='elu', kernel_initializer='he_normal'),
	tf.keras.layers.BatchNormalization(),
	tf.keras.layers.Dropout(0.2),
	tf.keras.layers.Dense(100, activation='elu', kernel_initializer='he_normal'),
	tf.keras.layers.BatchNormalization(),
	tf.keras.layers.Dropout(0.2),
	tf.keras.layers.Dense(100, activation='elu', kernel_initializer='he_normal'),
	tf.keras.layers.BatchNormalization(),
	tf.keras.layers.Dropout(0.2),
	tf.keras.layers.Dense(100, activation='elu', kernel_initializer='he_normal'),
	tf.keras.layers.BatchNormalization(),
	tf.keras.layers.Dropout(0.2),
	tf.keras.layers.Dense(10, activation='softmax'),
])


def train(model, x_train_, y_train_, x_test_, y_test_):
	model.compile(
		optimizer='adam',
		loss='sparse_categorical_crossentropy',
		metrics=['accuracy'],
	)

	# コールバック
	log_dir = "./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
	tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

	early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='auto')

	# 学習
	model.fit(
		x_train_,
		y_train_,
		epochs=50,
		validation_data=(x_test_, y_test_),
		callbacks=[tensorboard_callback, early_stopping_callback])

	save(model)


if __name__=='__main__':
	# train(model, x_train_0_4, y_train_0_4, x_test_0_4, y_test_0_4)
	model_dir = './models/20200329-215145'
	model = load(model_dir)
	print(model.summary())

	# モデルの凍結
	for layer in model.layers[:16]:
		layer.trainable = False
	# レイヤー入れ替え
	model.pop()
	model.pop()
	model.pop()
	model.pop()
	model.add(tf.keras.layers.Dense(100, activation='elu', kernel_initializer='he_normal', name='dense_4'))
	model.add(tf.keras.layers.BatchNormalization(name='batch_normalization_4'))
	model.add(tf.keras.layers.Dropout(0.2, name='dropout_4'))
	model.add(tf.keras.layers.Dense(10, activation='softmax', name='dense_5'))

	train(model, x_train_5_9, y_train_5_9, x_test_5_9, y_test_5_9)
