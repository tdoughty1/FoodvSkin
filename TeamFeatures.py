import skinmap as sm
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from scipy import ndimage, fftpack
from skimage import exposure, measure
import pandas as pd
from PIL import Image
import cStringIO
import urllib2
import numpy as np


# Get url
filePeople = 'People.txt'
urls = np.loadtxt(filePeople, dtype="str")
url = urls[1]

#
read = urllib2.urlopen(url).read()
obj = Image.open( cStringIO.StringIO(read) )
img = np.array(obj)
plt.imshow(img)
plt.show()
data = sm.SetUpImage(img)
print "image read in"

labels_df = pd.DataFrame(columns=['label_number', 'area_covered', 'density_cbcr', 'hwratio', 'density_hull', 'circularity'])

image= data.ImgReg.skin.cccm
#Define a 3x3 array with connectivity 2, in which all elements of the array are True
#http://docs.scipy.org/doc/scipy/reference/ndimage.html#module-scipy.ndimage.morphology
struct1 = ndimage.generate_binary_structure(2, 2)

#Dilate image
img_dilated = ndimage.morphology.binary_dilation(image, iterations=1, structure=struct1)
#Fill holes in the image
img_filled = ndimage.morphology.binary_fill_holes(img_dilated)
#Erode image
img_eroded = ndimage.morphology.binary_erosion(img_filled, iterations = 2, structure = struct1)

print "erosion done"

#Draw the different steps of image processing in one plot
img_gaussian = ndimage.gaussian_filter(img_eroded,1)
labeled, shapes = ndimage.label(img_gaussian > 0)

props = measure.regionprops(labeled)
img_size = (labeled.shape)[0]*(labeled.shape)[1]

print "measure.regionprops done"

for label in range(1, shapes+1):
    density = float(((labeled==label)&data.ImgReg.skin.cccm).sum())/float((labeled==label).sum())
    #defines a region of interest if at least 50% of its pixels are skin color, but less than 100%,
    #and if the area is bigger than an arbritary number
    print "in label loop"
    if density>0.5 and density<1 and props[label-1].area > 50: 
        density_cbcr = density
        area_covered = props[label-1].area/img_size
        hwratio = props[label-1].major_axis_length/props[label-1].minor_axis_length
        density_hull = props[label-1].solidity
        circularity = props[label-1].eccentricity
        labels_df.loc[len(labels_df.index)] = [label, area_covered, density_cbcr, hwratio, density_hull, circularity]

print "label features added to data frame"

# labels_df = pd.DataFrame(columns=['percent_skin','percent_skin_equalized','label_number', 'area_covered', 'density_cbcr', 'hwratio', 'density_hull', 'circularity'])

labels_df["percent_skin"]=np.nan
labels_df.percent_skin[0:3] = np.array([sm.percent_skin(data.ImgReg.skin.abg), \
                                      sm.percent_skin(data.ImgReg.skin.cbcr), \
                                      sm.percent_skin(data.ImgReg.skin.cccm)])

labels_df["percent_skin_equalized"]=np.nan
labels_df.percent_skin_equalized[0:3] = np.array([sm.percent_skin(data.ImgEq.skin.abg), \
                                      sm.percent_skin(data.ImgEq.skin.cbcr), \
                                      sm.percent_skin(data.ImgEq.skin.cccm)])

print "percent skin added to df"

labels_df.to_csv("Features.csv")
