# see README for reference links

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import random
from tensorflow.keras.applications import vgg19

CONTENT_IMG_PATH = './monke.jpg'
STYLE_IMG_PATH = './mosaic.jpg' 
NAME = 'monkmo'

total_weight = 1e-6
style_weight = 1e-6
content_weight = 2.5e-8

CONTENT_IMG_H = 500
CONTENT_IMG_W = 500

STYLE_IMG_H = 500
STYLE_IMG_W = 500

def preprocess_image(image_path):
    img = keras.preprocessing.image.load_img(image_path, target_size=(CONTENT_IMG_H, CONTENT_IMG_W))
    img = keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return tf.convert_to_tensor(img)


def deprocess_image(x):
    x = x.reshape((CONTENT_IMG_H, CONTENT_IMG_W, 3))

    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68

    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype("uint8")
    return x

def gram_matrix(x):
    x = tf.transpose(x, (2, 0, 1))
    features = tf.reshape(x, (tf.shape(x)[0], -1))
    gram = tf.matmul(features, tf.transpose(features))
    return gram

def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = CONTENT_IMG_H * CONTENT_IMG_W
    return tf.reduce_sum(tf.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))

def content_loss(base, combination):
    return tf.reduce_sum(tf.square(combination - base))

def total_loss(x):
    a = tf.square(
        x[:, : CONTENT_IMG_H - 1, : CONTENT_IMG_W - 1, :] - x[:, 1:, : CONTENT_IMG_W - 1, :]
    )
    b = tf.square(
        x[:, : CONTENT_IMG_H - 1, : CONTENT_IMG_W - 1, :] - x[:, : CONTENT_IMG_H - 1, 1:, :]
    )
    return tf.reduce_sum(tf.pow(a + b, 1.25))

# Build a VGG19 model loaded with pre-trained ImageNet weights
model = vgg19.VGG19(weights="imagenet", include_top=False)

# Get the symbolic outputs of important layers.
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

# Set up a model that returns the activation values for every layer in VGG19.
feature_extractor = keras.Model(inputs=model.inputs, outputs=outputs_dict)

style_layer_names = ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1",]

# The layer to use for the content loss.
content_layer_name = "block5_conv2"


def compute_loss(tData, cData, sData):
    input_tensor = tf.concat([cData, sData, tData], axis=0)
    features = feature_extractor(input_tensor)

    # Initialize the loss
    loss = tf.zeros(shape=())

    # Add content loss
    layer_features = features[content_layer_name]
    base_image_features = layer_features[0, :, :, :]
    combination_features = layer_features[2, :, :, :]
    loss = loss + content_weight * content_loss(base_image_features, combination_features)

    # Add style loss
    for layer_name in style_layer_names:
        layer_features = features[layer_name]
        style_reference_features = layer_features[1, :, :, :]
        combination_features = layer_features[2, :, :, :]
        sl = style_loss(style_reference_features, combination_features)
        loss += (style_weight / len(style_layer_names)) * sl

    # Add total variation loss
    loss += total_weight * total_loss(tData)
    return loss

@tf.function
def get_grads(tData, cData, sData):
    with tf.GradientTape() as tape:
        loss = compute_loss(tData, cData, sData)
    grads = tape.gradient(loss, tData)
    return loss, grads

def style_transfer(tData, cData, sData):
  optimizer = keras.optimizers.SGD(
      keras.optimizers.schedules.ExponentialDecay(
          initial_learning_rate=100.0, decay_steps=100, decay_rate=0.96
      )
  )

  iterations = 500
  for i in range(1, iterations + 1):
      loss, grads = get_grads(tData, cData, sData)
      optimizer.apply_gradients([(grads, tData)])
      if i % 100 == 0:
          print("Iteration %d: loss=%.2f" % (i, loss))
          img = deprocess_image(tData.numpy())
          fname = NAME + "_at_iteration_%d.png" % i
          keras.preprocessing.image.save_img(fname, img)



def main():
  cData = preprocess_image(CONTENT_IMG_PATH)
  sData = preprocess_image(STYLE_IMG_PATH)
  tData = tf.Variable(preprocess_image(CONTENT_IMG_PATH))
  style_transfer(tData, cData, sData)

if __name__ == '__main__':
  main()