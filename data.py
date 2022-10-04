from fastbook import *
from os import listdir
from numpy import asarray
from numpy import save
import numpy as np
from shutil import copyfile
import os

##### Download images ######
urls = search_images_ddg('vulture', max_images=40)
i = 0
while i < 40:
    try:
        download_url(urls[i], 'dataset_vultures_vs_sharks/vulture' +
                     str(i+1).zfill(2)+'.jpg')
        i += 1
    except:
        pass

i = 0
urls = search_images_ddg('shark', max_images=40)
while i < 40:
    try:
        download_url(urls[i], 'dataset_vultures_vs_sharks/shark' +
                     str(i+1).zfill(2)+'.jpg')
        i += 1
    except:
        pass

##### Define Labels ######

# define location of dataset
# folder = 'dataset_vultures_vs_sharks/'
# photos, labels = list(), list()
# # enumerate files in the directory
# for file in listdir(folder):
#     # determine class
#     output = 0.0
#     if file.startswith('vulture'):
#         output = 1.0
#     # load image
#     photo = load_img(folder + file, target_size=(200, 200))
#     # convert to numpy array
#     photo = img_to_array(photo)
#     # store
#     photos.append(photo)
#     labels.append(output)
# # convert to a numpy arrays
# photos = asarray(photos)
# labels = asarray(labels)
# print(photos.shape, labels.shape)
# # save the reshaped photos
# save('vultures_vs_sharks_photos.npy', photos)
# save('vultures_vs_sharks_labels.npy', labels)

##### Train Test Split ######

# create directories
dataset_home = 'preprocessed_dataset_vultures_vs_sharks/'
subdirs = ['train/', 'test/']
for subdir in subdirs:
    # create label subdirectories
    labeldirs = ['vultures/', 'sharks/']
    for labldir in labeldirs:
        newdir = dataset_home + subdir + labldir
        os.makedirs(newdir, exist_ok=True)


# seed random number generator
np.random.seed(1)
# define ratio of pictures to use for validation
val_ratio = 0.25
# copy training dataset images into subdirectories
src_directory = 'dataset_vultures_vs_sharks/'
arr = np.arange(40)
np.random.shuffle(arr)
i = 0
for file in listdir(src_directory):
    if file.startswith('vulture'):
        continue
    src = src_directory + '/' + file
    dst_dir = 'train/'
    if arr[i] < val_ratio*40:
        dst_dir = 'test/'
    print(file, i, dst_dir)
    dst = dataset_home + dst_dir + 'sharks/' + file
    copyfile(src, dst)
    i += 1
i = 0
for file in listdir(src_directory):
    if file.startswith('shark'):
        continue
    src = src_directory + '/' + file
    dst_dir = 'train/'
    if arr[i] < val_ratio*40:
        dst_dir = 'test/'
    print(file, i, dst_dir)
    dst = dataset_home + dst_dir + 'vultures/' + file
    copyfile(src, dst)
    i += 1
