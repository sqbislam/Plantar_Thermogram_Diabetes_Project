# -*- coding: utf-8 -*-
"""Further test VoxelMorph (All Images).ipynb

Original file is located at
    https://colab.research.google.com/drive/1vlrXfU9P4Gsy7IU_bnjQZfOzLCHpT27U

### References
[VoxelMorph at TMI](https://arxiv.org/abs/1809.05231)   
[Diffeomorphic VoxelMorph at MedIA](https://arxiv.org/abs/1903.03545)   
[Neurite Library](https://github.com/adalca/neuron) - [CVPR](http://arxiv.org/abs/1903.03148)

---
# Preamble
## Setup of environment
"""

# install voxelmorph, which will also install dependencies: neurite and pystrum

"""Common imports  """

from pathlib import Path
import pathlib
import shutil
import os


# imports
import os, sys

# third party imports
import numpy as np
import tensorflow as tf
assert tf.__version__.startswith('2.')

"""Next, we import two packages that will help us   
- [voxelmorph](http://voxelmorph.mit.edu) is a deep-learning based registration library  
- [neurite](https://github.com/adalca/neurite) is a library for medical image analysis with tensorflow  
"""

# local imports
import voxelmorph as vxm
import neurite as ne

from PIL import Image
import csv
import matplotlib.pyplot as plt
from utils import load_annotations, resize_with_padding, save_images_resized

"""
# Data
"""


DATA_PATH = "/content/drive/MyDrive/UM_MSC/Research_Thermal_Imaging/data/ThermoDataBase"
IMG_PATH = DATA_PATH 
TEST_PATH = IMG_PATH + "/Further_Test"
ALL_PATH = IMG_PATH + "/All"
TEST_IMAGE_PATH_L = IMG_PATH + "/All/L"
TEST_IMAGE_PATH_R = IMG_PATH + "/All/R"
TEST_IMAGE_PATH_SORTED = IMG_PATH + "/All/Updated/CompressedData/Train"
SORTED_IMG_PATH = IMG_PATH + "/All/Sorted"
GT_PATH = IMG_PATH + "/Further_Test/truth"
CSV_FILE_PATH = "/content/drive/MyDrive/UM_MSC/Research_Thermal_Imaging/data/ThermoDataBase/csvs/registration-voxel"

sub_path = ["", "/Augmented1", "/Augmented2"]
path = Path(TEST_IMAGE_PATH_SORTED + sub_path[0])
curr_path = TEST_IMAGE_PATH_SORTED + sub_path[0]


# Load images

# LEFT IMAGES
diabetic_images_L = np.array([np.asarray(Image.open("./DM_L/"+path_name).convert('L')) for path_name in sorted(os.listdir(curr_path + "/DM_L"))])
control_images_L = np.array([np.asarray(Image.open("./CG_L/"+path_name).convert('L')) for path_name in sorted(os.listdir(curr_path + "/CG_L"))])
all_images_L = np.append(diabetic_images_L, control_images_L, axis=0)

# RIGHT IMAGES
diabetic_images_R = np.array([np.asarray(Image.open("./DM_R/"+path_name).convert('L')) for path_name in sorted(os.listdir(curr_path + "/DM_R"))])
control_images_R = np.array([np.asarray(Image.open("./CG_R/"+path_name).convert('L')) for path_name in sorted(os.listdir(curr_path + "/CG_R"))])
all_images_R = np.append(diabetic_images_R, control_images_R, axis=0)

print(f'Diabetic Images: {diabetic_images_L.shape} Control Images: {control_images_L.shape}, All: {all_images_L.shape}')
print(f'Diabetic Images: {diabetic_images_R.shape} Control Images: {control_images_R.shape}, All: {all_images_R.shape}')

GT_PATH = "/content/drive/MyDrive/UM_MSC/Research_Thermal_Imaging/data/ThermoDataBase/All/Sorted/GT"
gt_images = np.array([np.asarray(Image.open(f"{GT_PATH}/" +path_name)) for path_name in sorted(os.listdir(GT_PATH))])


annotations_test = load_annotations("/content/drive/MyDrive/UM_MSC/Research_Thermal_Imaging/data/ThermoDataBase/All/Sorted/labels.csv")
annotations_test = annotations_test.reshape(2,7,2)


x_train = all_images_R[:115, :, :]
x_val = all_images_R[115:150, :, :]
x_test = all_images_R[150:, :, :]
print('shape of x_train: {}, x_val: {}, x_test: {}'.format(x_train.shape,x_val.shape, x_test.shape))

x_train = all_images_L[:115, :, :]
x_val = all_images_L[115:150, :, :]
x_test = all_images_L[150:, :, :]
print('shape of x_train: {}, x_val: {}, x_test: {}'.format(x_train.shape,x_val.shape, x_test.shape))

# Scale data
gt_images = gt_images.astype('float')/255
x_train = x_train.astype('float')/255
x_val = x_val.astype('float')/255
x_test = x_test.astype('float')/255
all_images_L_scaled = all_images_L.astype('float')/255
all_images_R_scaled = all_images_R.astype('float')/255
# verify
print('training maximum value', x_train.max())



"""**ML detour**: separating your data in *only* train/test **often leads to problems**   
You wouldn't want to iteratively (A) build a model, (B) train on training data, and (C) test on test data  
Doing so will **overfit to you test set** (because you will have adapted your algorithm to your test data). It's a common mistakes in ML submissions.  

We will split the 'training' into 'train/validate' data, and keep the test set for later  
And will only look at the test data at the very end (once we're ready to submit the paper!)


# CNN Model
"""

def buildModel(x_train):
  # configure unet input shape (concatenation of moving and fixed images)
  ndim = 2
  unet_input_features = 2
  inshape = (*x_train.shape[1:], unet_input_features)

  # configure unet features 
  nb_features = [
      [32, 32, 32, 32, 32],         # encoder features
      [32, 32, 32, 32, 16]  # decoder features
  ]

  # build model
  unet = vxm.networks.Unet(inshape=inshape, nb_features=nb_features)

  x_train.shape[1:]

  print('input shape: ', unet.input.shape)
  print('output shape:', unet.output.shape)

  # transform the results into a flow field.
  disp_tensor = tf.keras.layers.Conv2D(ndim, kernel_size=3, padding='same', name='disp')(unet.output)

  # check tensor shape
  print('displacement tensor:', disp_tensor.shape)

  """### Loss

  Given that the displacement $\phi$ is output from the network,  
  we need to figure out a loss to tell if it makes sense

  In a **supervised setting** we would have ground truth deformations $\phi_{gt}$,  
  and we could use a supervised loss like MSE $= \| \phi - \phi_{gt} \|$

  The main idea in **unsupervised registration** is to use loss inspired by classical registration  

  Without supervision, how do we know this deformation is good?  
  (1) make sure that $m \circ \phi$ ($m$ warped by $\phi$) is close to $f$  
  (2) regularize $\phi$ (often meaning make sure it's smooth)

  To achieve (1), we need to *warp* input image $m$. To do this, we use a spatial transformation network layer, which essentially does linear interpolation.
  """

  # build transformer layer
  spatial_transformer = vxm.layers.SpatialTransformer(name='transformer')

  # extract the first frame (i.e. the "moving" image) from unet input tensor
  moving_image = tf.expand_dims(unet.input[..., 0], axis=-1)

  # warp the moving image with the transformer
  moved_image_tensor = spatial_transformer([moving_image, disp_tensor])

  outputs = [moved_image_tensor, disp_tensor]
  vxm_model = tf.keras.models.Model(inputs=unet.inputs, outputs=outputs)

  # build model using VxmDense
  inshape = x_train.shape[1:]
  vxm_model = vxm.networks.VxmDense(inshape, nb_features, int_steps=0)

  print('input shape: ', ', '.join([str(t.shape) for t in vxm_model.inputs]))
  print('output shape:', ', '.join([str(t.shape) for t in vxm_model.outputs]))

  # voxelmorph has a variety of custom loss classes
  losses = [vxm.losses.NCC().loss, vxm.losses.Grad('l1').loss]

  # usually, we have to balance the two losses by a hyper-parameter
  lambda_param = 0.17
  loss_weights = [1, lambda_param]

  vxm_model.compile(optimizer='Adam', loss=losses, loss_weights=loss_weights)
  return vxm_model



vxm_model = buildModel(x_train)

from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
import math 
import datetime


"""# Train Model

To train, we need to make sure the data is in the right format and fed to the model the way we want it keras models can be trained with `model.fit`, which requires all the data to be in a big array, or `model.fit_generator`, which requires a python generator that gives you batches of data.

Let's code a simple data generator based on the MNIST data.
"""

def vxm_data_generator(x_data, batch_size=1, log=False, index=0, LorR = 0):
    """
    Generator that takes in data of size [N, H, W], and yields data for
    our custom vxm model. Note that we need to provide numpy data for each
    input, and each output.

    LorR = Left 0 Right 1

    inputs:  moving [bs, H, W, 1], fixed image [bs, H, W, 1]
    outputs: moved image [bs, H, W, 1], zero-gradient [bs, H, W, 2]
    """

    # preliminary sizing
    vol_shape = x_data.shape[1:] # extract data shape
    ndims = len(vol_shape)
    
    # prepare a zero array the size of the deformation
    # we'll explain this below
    zero_phi = np.zeros([batch_size, *vol_shape, ndims])
    
    while True:
        # prepare inputs:
        # images need to be of the size [batch_size, H, W, 1]
        idx1 = np.random.randint(0, x_data.shape[0], size=batch_size)
        if(index > 0):
          idx1 = [index]
        moving_images = x_data[idx1, ..., np.newaxis]
        # idx2 = np.random.randint(0, x_data.shape[0], size=batch_size)
        # Left = 0, Right = 1
        idx2 = [LorR] # First image is always template
        # idx2 = np.random.randint(0, gt_images.shape[0], size=batch_size)
        fixed_images = gt_images[idx2, ..., np.newaxis]
        inputs = [moving_images, fixed_images]
        
        
        # prepare outputs (the 'true' moved image):
        # of course, we don't have this, but we know we want to compare 
        # the resulting moved image with the fixed image. 
        # we also wish to penalize the deformation field. 
        if(log == True):
           outputs = [fixed_images, zero_phi, idx1[0]]
        else:
          outputs = [fixed_images, zero_phi]
        
        yield (inputs, outputs)



# Callbacks while training
def get_callbacks(model_file, initial_learning_rate=0.005, learning_rate_drop=0.3, learning_rate_epochs=None,
                  learning_rate_patience=50, logging_file=f"{CSV_FILE_PATH}/training.log", verbosity=1,
                  early_stopping_patience=None):
    callbacks = list()
    callbacks.append(ModelCheckpoint(filepath=CSV_FILE_PATH+'/model_reg_right.{epoch:02d}-{loss:.2f}.h5',save_best_only=True, mode="min", monitor='loss'))
    callbacks.append(CSVLogger(logging_file, append=True))
    log_dir = CSV_FILE_PATH + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    # callbacks.append(tensorboard_callback)

    # Create a callback that saves the model's weights every 5 epochs
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
      filepath=CSV_FILE_PATH+"/cp_right{epoch:03d}.ckpt", 
      verbose=0, 
      save_weights_only=True,
      save_freq=10)
    callbacks.append(cp_callback)
    callbacks.append(ReduceLROnPlateau(factor=learning_rate_drop, patience=learning_rate_patience,
                                           verbose=verbosity,monitor="loss"))
    if early_stopping_patience:
        callbacks.append(EarlyStopping(monitor="val_accuracy", verbose=verbosity, patience=early_stopping_patience, min_delta=0.1))
    return callbacks

# let's test it
train_generator = vxm_data_generator(x_train)
in_sample, out_sample = next(train_generator)

# visualize
images = [img[0, :, :, 0] for img in in_sample + out_sample] 
titles = ['moving', 'fixed', 'moved ground-truth (fixed)', 'zeros']
ne.plot.slices(images, titles=titles, cmaps=['gray'], do_colorbars=True)

inp, out = next(train_generator)

# nb_epochs = 200
# steps_per_epoch = 190
# hist = vxm_model.fit_generator(train_generator, epochs=nb_epochs, steps_per_epoch=steps_per_epoch,callbacks=get_callbacks(f"{CSV_FILE_PATH}/regist-r"), verbose=2);

# Save model weights
# vxm_model.save_weights(f"{CSV_FILE_PATH}/final-registr-right.h5")


# Load model weights
vxm_model.load_weights(f"{CSV_FILE_PATH}/final-registr-left.h5")

import joblib
# Save model
joblib.dump(vxm_model, f'{CSV_FILE_PATH}/registration_model.pkl')

"""It's always a good idea to visualize the loss, not just read off the numbers. """

def plot_history(hist, loss_name='loss'):
    # Simple function to plot training history.
    plt.figure()
    plt.plot(hist.epoch, hist.history[loss_name], '.-')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()


"""# Registration

With pair-wise optimization methods (like most classical methods), to register a new pair you would need to optimize a deformation field.  

With learning based registration, we simply evaluate the network for a new input pair
"""

global_mean = []
local_mean = []

from skimage.metrics import structural_similarity as ssim
def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err
def compare_images(imageA, imageB, title):
	# compute the mean squared error and structural similarity
	# index for the images
	m = mse(imageA, imageB)
	s = ssim(imageA, imageB)
	# setup the figure
	fig = plt.figure(title)
	plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))
	# show first image
	ax = fig.add_subplot(1, 2, 1)
	plt.imshow(imageA, cmap = plt.cm.gray)
	plt.axis("off")
	# show the second image
	ax = fig.add_subplot(1, 2, 2)
	plt.imshow(imageB, cmap = plt.cm.gray)
	plt.axis("off")
	# show the images
	plt.show()

def l1_loss(y_prime, y):
  l1 = np.sqrt(np.power((y-y_prime),2))
  return l1

def get_mean(x, y):
  mean = []
  for _, (pred, truth) in enumerate(zip(x,y)):
    mean.append([l1_loss(pred[0], truth[0]), l1_loss(pred[1], truth[1])])
  return mean

from matplotlib.colors import Normalize
def plot_histogram(a, title):
  plt.figure()
  fig, ax = plt.subplots(figsize=(10,4))
  n,bins,patches = ax.hist(a, edgecolor='none')
  ax.set_title("histogram")
  ax.set_xlim(0,1)

  plt.title(title)
  cm = plt.cm.get_cmap('cool')
  norm = Normalize(vmin=bins.min(), vmax=bins.max())
  
  plt.show()

from scipy import signal
def get_corr(im1, im2):
  corr = signal.correlate2d(im1, im2, boundary='symm', mode='same')

  plt.figure()
  plt.imshow(corr, cmap='gray')
  plt.figure()

root_dir = curr_path + "/Annotations"
pred_dir = curr_path + "/Images/"
DM_L_dir = root_dir + "/DM_L/"
DM_R_dir = root_dir + "/DM_R/"
CG_L_dir = root_dir + "/CG_L/"
CG_R_dir = root_dir + "/CG_R/"


diabetic_L_scaled = diabetic_images_L.astype('float')/255
diabetic_R_scaled = diabetic_images_R.astype('float')/255
control_L_scaled = control_images_L.astype('float')/255
control_R_scaled = control_images_R.astype('float')/255

x_train[0].shape

def get_random_pred(index, data, dir, pred_dir, save=False, hist=False):
  # let's get some data
  val_generator = vxm_data_generator(data, batch_size = 1, log=True, index = index)
  val_input, out = next(val_generator)
  _, zero, val_idx = out
  val_pred = vxm_model.predict(val_input)

  # visualize
  images = [img[0, :, :, 0] for img in val_input + val_pred] 
  titles = ['moving', 'fixed', 'moved', 'flow']
  
  # Plot at 10 inages intervals
  if(index % 10 == 0):
    ne.plot.slices(images, titles=titles, cmaps=['gray'], do_colorbars=True,show=True);


  # Get outputs
  moving = images[0]
  fixed = images[1]
  moved = images[2]
  annotations = annotations_test[1]

  # get dense field of inverse affine
  field_inv = val_pred[1].squeeze()
  field_inv = field_inv[np.newaxis, ...]
  annotations_keras = annotations[np.newaxis, ...]

  # warp annotations
  data = [tf.convert_to_tensor(f, dtype=tf.float32) for f in [annotations_keras, field_inv]]
  annotations_warped = vxm.utils.point_spatial_transformer(data)[0, ...].numpy()

  plt.figure(figsize=(10, 6))

  # Save Figures 
  if(save == True):
    # Fixed
  
    if(index == 0):
      #plt.savefig(dir+"Images/"+"fixed"+str(index), bbox_inches="tight")
      np.save(dir + "fixed"+str(index)+".npy", fixed, allow_pickle=True, fix_imports=True)
      np.save(dir + "fixed-annotations"+str(index)+".npy", annotations, allow_pickle=True, fix_imports=True)
    
    # Save every 10 images
    if(index % 10 == 0):
      # New Prediciton
      plt.figure()
      plt.subplot(1,2,1)
      plt.imshow(fixed, cmap='gray')
      plt.plot(*[annotations[:, f] for f in [0, 1]], 'o',color="red")  
      plt.title("Template Image \n (with annotations)")
      
      plt.subplot(1,2,2)
      plt.imshow(moving, cmap='gray')
      plt.plot(*[annotations_warped[:, f] for f in [0, 1]], 'o', color="red")
      plt.title("New Image \n (with predicted annotations)")
    
      plt.savefig(pred_dir + "pred"+str(index), bbox_inches="tight")
    
    # np.save(dir+"annotations/"+"pred"+str(index)+".npy", moving, allow_pickle=True, fix_imports=True)
    np.save(dir + "pred-annotations"+str(index)+".npy", annotations_warped, allow_pickle=True, fix_imports=True)



get_random_pred(index = 20,data=diabetic_L_scaled,dir =DM_L_dir, pred_dir=pred_dir, save=True)

# for i in range(0,len(diabetic_L_scaled)):
#   get_random_pred(index = i, data=diabetic_L_scaled, dir=DM_L_dir, pred_dir = pred_dir, save=True)
# for i in range(0,len(diabetic_R_scaled)):
#   get_random_pred(index = i, data=diabetic_R_scaled, dir=DM_R_dir, pred_dir = pred_dir, save=True)
# for i in range(0, len(control_L_scaled)):
#   get_random_pred(index = i, data=control_L_scaled, dir=CG_L_dir, pred_dir = pred_dir,save=True)
# for i in range(0, len(control_L_scaled)):
#   get_random_pred(index = i, data=control_R_scaled, dir=CG_R_dir, pred_dir = pred_dir,save=True)

