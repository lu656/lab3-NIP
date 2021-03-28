
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import random
# from scipy.misc import imsave, imresize
# from scipy.optimize import fmin_l_bfgs_b   # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html
from tensorflow.keras.applications import vgg19
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# import warnings

# tf.compat.v1.disable_eager_execution()

# from __future__ import print_function

# pytorch things
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim # for efficient gradient descent

# load/display images
from PIL import Image 
# from IPython.display import Image, display
import matplotlib.pyplot as plt

# import torchvision.transforms as transforms # transform PIL images into tensors
# import torchvision.models as models # train or load pretrained models

import copy


random.seed(1618)
np.random.seed(1618)
#tf.set_random_seed(1618)   # Uncomment for TF1.
tf.random.set_seed(1618)

#tf.logging.set_verbosity(tf.logging.ERROR)   # Uncomment for TF1.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# CONTENT_IMG_PATH = "./dancing.jpg"           #TODO: Add this.
# STYLE_IMG_PATH = "./picasso.jpg"             #TODO: Add this.
CONTENT_IMG_PATH = keras.utils.get_file('paris.jpg', 'https://i.imgur.com/F28w3Ac.jpg')
STYLE_IMG_PATH = keras.utils.get_file('starry_night.jpg', 'https://i.imgur.com/9ooB60I.jpg')             #TODO: Add this.



CONTENT_IMG_H = 500
CONTENT_IMG_W = 500

STYLE_IMG_H = 500
STYLE_IMG_W = 500

CONTENT_WEIGHT = 2.5e-8    # Alpha weight. 0.1
STYLE_WEIGHT = 1e-6      # Beta weight. 1.0
TOTAL_WEIGHT = 1e-6    # 1.0

TRANSFER_ROUNDS = 3


#=============================<Helper Fuctions>=================================
'''
TODO: implement this.
This function should take the tensor and re-convert it to an image.
'''
def deprocessImage(img):
    img = img.reshape((CONTENT_IMG_W, CONTENT_IMG_H, 3)) # convert tensor into valid image
    # remove zero-cener by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # transform image pixels to be within range 0-255
    img = img[:, :, ::-1]
    img = np.clip(img, 0, 255).astype('uint8')
    return img

# used to compute style loss
def gramMatrix(x):
    # features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    # gram = K.dot(features, K.transpose(features))
    x = tf.transpose(x, (2, 0, 1))
    features = tf.reshape(x, (tf.shape(x)[0], -1))
    gram = tf.matmul(features, tf.transpose(features))
    return gram



#========================<Loss Function Builder Functions>======================

# keeps generated image close to local textures in style image
def styleLoss(style, gen):
    # return None   #TODO: implement.
    S = gramMatrix(style)
    C = gramMatrix(gen)
    channels = 3
    size = CONTENT_IMG_H * CONTENT_IMG_W
    return tf.reduce_sum(tf.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2)) 

# keeps high-level representation of generated image close to base image
def contentLoss(content, gen):
    return tf.reduce_sum(tf.square(gen - content))

# keeps generated image locally-coherent
def totalLoss(x):
    # # return None   #TODO: implement.
    # a = tf.square(x[:, : CONTENT_IMG_H - 1, : CONTENT_IMG_W - 1, :] - x[:, 1:, : CONTENT_IMG_W - 1, :])
    # b = tf.square(x[:, : CONTENT_IMG_H - 1, : CONTENT_IMG_W - 1, :] - x[:, : CONTENT_IMG_W - 1, 1:, :])
    # return tf.reduce_sum(tf.pow(a + b, 1.25))
    a = tf.square(x[:, : CONTENT_IMG_H - 1, : CONTENT_IMG_W - 1, :] - x[:, 1:, : CONTENT_IMG_W - 1, :])
    b = tf.square(x[:, : CONTENT_IMG_H - 1, : CONTENT_IMG_W - 1, :] - x[:, : CONTENT_IMG_H - 1, 1:, :])
    return tf.reduce_sum(tf.pow(a + b, 1.25))





#=========================<Pipeline Functions>==================================

def getRawData():
    print("   Loading images.")
    print("      Content image URL:  \"%s\"." % CONTENT_IMG_PATH)
    print("      Style image URL:    \"%s\"." % STYLE_IMG_PATH)
    # cImg = load_img(CONTENT_IMG_PATH)
    # tImg = cImg.copy()
    # sImg = load_img(STYLE_IMG_PATH)
    # cImg = Image.open(CONTENT_IMG_PATH)
    # tImg = copy.deepcopy(cImg)
    # sImg = Image.open(STYLE_IMG_PATH)
    cImg = CONTENT_IMG_PATH
    tImg = CONTENT_IMG_PATH
    sImg = STYLE_IMG_PATH
    print("      Images have been loaded.")
    return ((cImg, CONTENT_IMG_H, CONTENT_IMG_W), (sImg, STYLE_IMG_H, STYLE_IMG_W), (tImg, CONTENT_IMG_H, CONTENT_IMG_W))

def preprocessData(raw):
    tmp, ih, iw = raw
    img = keras.preprocessing.image.load_img(tmp, target_size=(ih, iw))
    # img = img.resize((ih, iw))
    img = keras.preprocessing.image.img_to_array(img)
    # img = img.astype('float32')
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return tf.convert_to_tensor(img)

# def preprocessData(raw):
#     img, ih, iw = raw
#     img = img_to_array(img)
#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore")
#         # img = imresize(img, (ih, iw, 3))
#         img = img.resize((ih, iw, 3))
#     img = img.astype("float64")
#     img = np.expand_dims(img, axis=0)
#     img = vgg19.preprocess_input(img)
#     return img

# def getRawData():
#     print("   Loading images.")
#     print("      Content image URL:  \"%s\"." % CONTENT_IMG_PATH)
#     print("      Style image URL:    \"%s\"." % STYLE_IMG_PATH)
#     loader = transforms.Compose([
#         transforms.Resize(CONTENT_IMG_H),
#         transforms.toTensor()
#     ])
#     cImg = Image.open(CONTENT_IMG_PATH)
#     cImg = loader(cImg).unsqueeze(0)
#     cImg = cImg.to(device, torch.float)


# builds vgg19 model with pre-trained ImageNet weights
model = vgg19.VGG19(weights='imagenet', include_top=False)
# save symbolic outputs of each "key" layer
outputDict = dict([(layer.name, layer.output) for layer in model.layers])
# set up model to return activation values for every layer in vgg19
feature_extractor = keras.Model(inputs=model.inputs, outputs=outputDict)
print("   VGG19 model loaded.")

styleLayerNames = ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1"]
contentLayerName = "block5_conv2"

def compute_loss(cData, sData, tData):
    print("   Building transfer model.")
    # contentTensor = K.variable(cData)
    # styleTensor = K.variable(sData)
    # genTensor = K.placeholder((1, CONTENT_IMG_H, CONTENT_IMG_W, 3))
    # inputTensor = K.concatenate([contentTensor, styleTensor, genTensor], axis=0)
    inputTensor = tf.concat([cData, sData, tData], axis=0)
    print('-------------------------------------------------------')
    print(inputTensor)
    print('=====================================================')

    # model = None   #TODO: implement.

    loss = tf.zeros(shape=())

    features = feature_extractor(inputTensor)

    print("   Calculating content loss.")
    contentLayer = features[contentLayerName]
    # contentFeatures = features[contentLayerName]
    contentOutput = contentLayer[0, :, :, :]
    genOutput = contentLayer[2, :, :, :]
    # loss += None   #TODO: implement.
    loss = loss + CONTENT_WEIGHT * contentLoss(contentOutput, genOutput)
    print("   Calculating style loss.")
    for layerName in styleLayerNames:
        # loss += None   #TODO: implement.
        styleLayer = features[layerName]
        styleOutput = styleLayer[1, :, :, :]
        genOutput = styleLayer[2, :, :, :]
        style_loss = styleLoss(styleOutput, genOutput)
        loss += (STYLE_WEIGHT / len(styleLayerNames)) * style_loss

    print('   Calculating total loss.')    
    # loss += None   #TODO: implement.

    loss += TOTAL_WEIGHT * totalLoss(tData)
    return loss
    # TODO: Setup gradients or use K.gradients().

@tf.function
def get_grads(cData, sData, tData):
    with tf.GradientTape() as tape:
        loss = compute_loss(cData, sData, tData)
    # print('===============================================')    
    # print(type(tape), type(loss), type(tData))
    # print('===============================================')    
    grads = tape.gradient(loss, tData)
    # loss = compute_loss(cData, sData, tData)
    # grads = K.gradients(loss, tData)
    print('grads is:', grads)
    print('loss is:', loss)
    return loss, grads

'''
TODO: Allot of stuff needs to be implemented in this function.
First, make sure the model is set up properly.
Then construct the loss function (from content and style loss).
Gradient functions will also need to be created, or you can use K.Gradients().
Finally, do the style transfer with gradient descent.
Save the newly generated and deprocessed images.
'''
def styleTransfer(cData, sData, tData):
    
    optimizer = keras.optimizers.SGD(
        keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=100, decay_steps=100, decay_rate=0.96
        )
    )

    print("   Beginning transfer.")
    for i in range(TRANSFER_ROUNDS):
        print("   Step %d." % i)
        #TODO: perform gradient descent using fmin_l_bfgs_b.
        tLoss, grads = get_grads(cData, sData, tData)
        print('----------------------------------------------------')
        print(grads)
        print('----------------------------------------------------')
        # fmin_l_bfgs_b(grads, tData)
        optimizer.apply_gradients([(grads, tData)])
        print("      Loss: %f." % tLoss)
        print(type(loss))
        img = deprocessImage(tData.numpy())
        plt.figure()
        plt.imshow(img)
        # saveFile = './iteration_at_%d.png' % i   #TODO: Implement.
        # imsave(saveFile, img)   #Uncomment when everything is working right.
        print("      Image saved to \"%s\"." % saveFile)
    print("   Transfer complete.")




#=========================<Main>================================================

def main():
    print("Starting style transfer program.")
    raw = getRawData()
    cData = preprocessData(raw[0])   # Content image.
    sData = preprocessData(raw[1])   # Style image.
    tData = tf.Variable(preprocessData(raw[2]))   # Transfer image.

    # print(cData)    
    # print("\n\n\n\n")
    # print(sData)
    # print("\n\n\n\n")
    # print(tData)
    # tData = tf.cast(tData, tf.int32)
    styleTransfer(cData, sData, tData)
    print("Done. Goodbye.")



if __name__ == "__main__":
    main()
