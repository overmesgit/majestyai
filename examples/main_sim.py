import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

target_coord = keras.Input(shape=(1,))

x = layers.concatenate([target_coord])
initializer = tf.keras.initializers.RandomNormal(mean=.5, stddev=.5)
coord1 = layers.Dense(2, activation="relu", kernel_initializer=initializer)(x)
coord2 = layers.Dense(2, activation="relu", kernel_initializer=initializer)(coord1)
coord3 = layers.Dense(2, activation="relu", kernel_initializer=initializer)(coord2)
x_left = layers.concatenate([coord3])
x_right = layers.concatenate([coord3])
left_output = layers.Dense(1, name="left_output", activation='sigmoid',
                           bias_initializer='ones')(x_left)
right_output = layers.Dense(1, name="right_output", activation='sigmoid',
                            bias_initializer='ones')(x_right)

model = keras.Model(inputs=[target_coord], outputs=[left_output, right_output],
                    name="model")
model.summary()
# keras.utils.plot_model(model, "multi_input_and_output_model.png", show_shapes=True)

x_train = np.array([
    (1, 0, 1),
    (3, 0, 1),
    (1, 0, 1),
    (2, 0, 1),
    (3, 0, 1),
    (5, 0, 1),
    (6, 0, 1),
    (7, 0, 1),
    (8, 0, 1),

    (-1, 1, 0),
    (-3, 1, 0),
    (-6, 1, 0),
    (-7, 1, 0),
    (-4, 1, 0),
    (-2, 1, 0),
    (-3, 1, 0),
    (-4, 1, 0),
])

data1 = x_train[:, [0, ]]
data3 = x_train[:, [1, ]]
data4 = x_train[:, [2, ]]

print(data1)
print(data3)
print(data4)

model.compile(
    loss=tf.keras.losses.Poisson(reduction="auto", name="poisson"),
    optimizer=keras.optimizers.RMSprop(learning_rate=0.1),
    metrics=["accuracy"],
)

history = model.fit([data1], [data3, data4],
                    shuffle=True, epochs=10, validation_split=0.2)

print(model.predict([np.array([10])]))
print(model.predict([np.array([50])]))
print(model.predict([np.array([-10])]))
print(model.predict([np.array([-20])]))

# test_scores = model.evaluate(x_test, y_test, verbose=2)
# print("Test loss:", test_scores[0])
# print("Test accuracy:", test_scores[1])
