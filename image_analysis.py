import cv2
import numpy as np
from scipy.stats import skew
from scipy.stats import entropy
import diplib as dip
from scipy.stats import linregress
import pandas
import math
from PIL import Image
import os
import csv

from scipy import ndimage

import torch
from PIL import Image
import torchvision.transforms as transforms

def get_image_gray(img_file_path):
    image = cv2.imread(img_file_path, cv2.IMREAD_UNCHANGED)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY )
    return gray

def get_gray_list(gray):
    list_gray = gray.ravel().tolist()
    return list_gray

def calculate_mean(list_gray):
    return np.mean(list_gray)

def calculate_std(list_gray):
    return np.std(list_gray)

def calculate_skewness(list_gray):
    return skew(list_gray)


# calculate the entropy
def calculate_entropy(gray_img):
    _bins = 128
    hist, _ = np.histogram(gray_img.ravel(), bins=_bins, range=(0, _bins))
    prob_dist = hist / hist.sum()
    return entropy(prob_dist, base=2)

# calculate the magnitude slope
# code reference https://www.geeksforgeeks.org/how-to-find-the-fourier-transform-of-an-image-using-opencv-python/

def calculate_magnitude_slope(gray_image):
    # Compute the discrete Fourier Transform of the image
    fourier = cv2.dft(np.float32(gray_image), flags=cv2.DFT_COMPLEX_OUTPUT)

    # Shift the zero-frequency component to the center of the spectrum
    fourier_shift = np.fft.fftshift(fourier)

    # calculate the magnitude of the Fourier Transform
    magnitude = 20 * np.log(cv2.magnitude(fourier_shift[:, :, 0], fourier_shift[:, :, 1]))

    # Scale the magnitude for display
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

    # Display the magnitude of the Fourier Transform
    # cv2.imshow('Fourier Transform', magnitude)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # radical profile reference https://stackoverflow.com/questions/21242011/most-efficient-way-to-calculate-radial-profile
    # center = (magnitude.shape[1]/2, magnitude.shape[0]/2)
    RADI = 50
    rad = dip.RadialMean(magnitude, binSize=1)
    # rad[RADI:].Show()
    radical_array = np.array(rad[RADI:])
    radical_index = [i for i in range(1, len(radical_array)+1)]
    magnitude_slope = linregress(radical_index, radical_array).slope

    return magnitude_slope



# fractal dimension
# code reference https://trvrm.github.io/fractal-dimension.html

#  create simple black and white images.
def bw(img):
    gray = img.convert('L')
    return gray.point(lambda x: 0 if x<128 else 1, '1')



# At various different scales, I want to divide each image up into squares and then count how many squares have at least one black pixel in them.
def interesting(image):
    #true if any data is 0, i.e. black
    return 0 in set(image.getdata())

# This function chops an image up into
def interesting_box_count(image, length):
    width,height=image.size

    interesting_count=0
    box_count=0
    for x in range(int(width/length)):
        for y in range(int(height/length)):
            C=(x*length,y*length,length*(x+1),length*(y+1))

            chopped = image.crop(C)
            box_count+=1
            if (interesting(chopped)):
                interesting_count+=1

    assert box_count
    assert interesting_count
    return interesting_count

# This returns pairs of numbers. One represents the scale, the other the (log) count of boxes at that scale that have black pixels in them.
def getcounts(image):
    length=min(image.size)
    while(length>5):
        interesting = interesting_box_count(image,length)
        yield math.log(1.0/length), math.log(interesting)
        length=int(length/2)



def calculate_fractal(image_path):
    map_image = bw(Image.open(img_file_path))
    counts = getcounts(map_image)
    frame = pandas.DataFrame(counts,columns=["x","y"])
    Fractal_D = linregress(frame.x,frame.y).slope
    return Fractal_D



richness_collection = []

img_folder_path = './Maps/'
# img_folder_path = './BingMaps/'

# read the file names from the file list
files = os.listdir(img_folder_path)
# removing the '.DS_Store' from Mac
if '.DS_Store' in files: files.remove('.DS_Store')
if 'desktop.ini' in files: files.remove('desktop.ini')
if 'Thumbs.db' in files: files.remove('Thumbs.db')

for f in files:
    img_file_path = img_folder_path + f
    print("Calculating file: " + f )

    # read the image
    gray_img = get_image_gray(img_file_path)
    gray_img_list = get_gray_list(gray_img)

    dict_variables = {}
    # calculate the richness
    dict_variables['id'] = f.split('.')[0]
    dict_variables['mean'] = calculate_mean(gray_img_list)
    dict_variables['std'] = calculate_std(gray_img_list)
    dict_variables['skewness'] = calculate_skewness(gray_img_list)
    dict_variables['entropy'] = calculate_entropy(gray_img)
    dict_variables['magnitude_slope'] = calculate_magnitude_slope(gray_img)
    dict_variables['fractal'] = calculate_fractal(img_file_path)

    richness_collection.append(dict_variables)

keys = richness_collection[0].keys()
with open('richness2.csv', 'w', newline='') as output_file:
    dict_writer = csv.DictWriter(output_file, keys)
    dict_writer.writeheader()
    dict_writer.writerows(richness_collection)

print('-----end-------')

