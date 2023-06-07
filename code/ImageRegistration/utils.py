from PIL import Image
import csv
import matplotlib.pyplot as plt
import os
import numpy as np

def load_annotations(file):
  '''
    Takes a file and retruns annotations

    Input:
      file : path to csv file
    Output:
      annotations numpy array
  '''
  annot = []
  if not os.path.exists(file):
      return []
  with open(file, 'r') as f:
      reader = csv.reader(f)
      for idx, line in enumerate(reader):
          label = line[-1]
          # strip BOM. \ufeff for python3,  \xef\xbb\bf for python2
          line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in line]
          label, x, y, name, w, h = list(line[:6])
          annot.append([x, y]) 
      return np.asarray(annot, dtype=np.float32)

def resize_with_padding(im, desired_size):
  '''
    Takes an image and resizes it to desired size

    Input:
      im : Image file
    Output:
      New image with desired size
  '''
  old_size = im.size  # old_size[0] is in (width, height) format

  ratio = float(desired_size)/max(old_size)
  new_size = tuple([int(x*ratio) for x in old_size])
  # use thumbnail() or resize() method to resize the input image

  # thumbnail is a in-place operation

  # im.thumbnail(new_size, Image.ANTIALIAS)
  im = im.resize(new_size, Image.ANTIALIAS)

  # create a new image and paste the resized on it
  new_im = Image.new("L", (desired_size, desired_size))
  new_im.paste(im, ((desired_size-new_size[0])//2,
                      (desired_size-new_size[1])//2))
  return new_im


def save_images_resized(FILE_PATH, DEST_FILE_PATH):
    """
        Save images to destination path after resizing
    
    """
    file_names = sorted(os.listdir(FILE_PATH))
    for path_name in file_names:
        resized_cropped_img = resize_with_padding(Image.open(path_name).convert('L'), 128)

        if(path_name.startswith("DM")):
            image_path = DEST_FILE_PATH + "/DM_R"
            image = resized_cropped_img.save(f"{image_path}/{path_name}")
        else:
            image_path = DEST_FILE_PATH + "/CG_R"
            image = resized_cropped_img.save(f"{image_path}/{path_name}")

