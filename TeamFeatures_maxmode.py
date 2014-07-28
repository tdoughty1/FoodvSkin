import skinmap as sm
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from scipy import ndimage, fftpack, stats
from skimage import exposure, measure
from pandas import DataFrame
from PIL import Image
import cStringIO
from urllib2 import urlopen, HTTPError
import numpy as np

# Get url
file = 'Food.txt'
urls = np.loadtxt(file, dtype="str")

labels_df=DataFrame()
labels_df['URL'] = urls
labels_df['percent_skin_abg'] = -1*np.ones(len(urls))
labels_df['percent_skin_cbcr'] = -1*np.ones(len(urls))
labels_df['percent_skin_cccm'] = -1*np.ones(len(urls))
labels_df['percent_skin_equalized_abg'] = -1*np.ones(len(urls))
labels_df['percent_skin_equalized_cbcr'] = -1*np.ones(len(urls))
labels_df['percent_skin_equalized_cccm'] = -1*np.ones(len(urls))
labels_df['label_number'] = -1*np.ones(len(urls))
labels_df['area_covered_max'] = -1*np.ones(len(urls))
labels_df['area_covered_mode'] = -1*np.ones(len(urls))
labels_df['density_max'] = -1*np.ones(len(urls))
labels_df['density_mode'] = -1*np.ones(len(urls))
labels_df['hwratio_max'] = -1*np.ones(len(urls))
labels_df['hwratio_mode'] = -1*np.ones(len(urls))
labels_df['density_hull_max'] = -1*np.ones(len(urls))
labels_df['density_hull_mode'] = -1*np.ones(len(urls))
labels_df['circularity_max'] = -1*np.ones(len(urls))
labels_df['circularity_mode'] = -1*np.ones(len(urls))

index = -1
for url in urls:
    
    index += 1

    try:
        read = urlopen(url).read()
        obj = Image.open(cStringIO.StringIO(read))
        img = np.array(obj)
        plt.imshow(img)
        plt.show()
        data = sm.SetUpImage(img)
        print "image read in"
        print url

        image = data.ImgReg.skin.cbcr
        #Define a 3x3 array with connectivity 2, in which all elements of the array are True
        #http://docs.scipy.org/doc/scipy/reference/ndimage.html#module-scipy.ndimage.morphology
        struct1 = ndimage.generate_binary_structure(2, 2)

        #Dilate image
        img_dilated = ndimage.morphology.binary_dilation(image, iterations=1, structure=struct1)
        #Fill holes in the image
        img_filled = ndimage.morphology.binary_fill_holes(img_dilated)
        #Erode image
        img_eroded = ndimage.morphology.binary_erosion(img_filled, iterations = 2, structure = struct1)
    
        #print "erosion done"

        #Draw the different steps of image processing in one plot
        img_gaussian = ndimage.gaussian_filter(img_eroded,1)
        labeled, shapes = ndimage.label(img_gaussian > 0)

        props = measure.regionprops(labeled)
        img_size = (labeled.shape)[0]*(labeled.shape)[1]

        print "measure.regionprops done"

        density_cbcr = np.array([])
        area_covered = np.array([])
        hwratio = np.array([])
        density_hull = np.array([])
        circularity = np.array([])
    
        for label in range(1, shapes+1):
            density = float(((labeled==label)*data.ImgReg.skin.cbcr).sum())/float((labeled==label).sum())
            # defines a region of interest if at least 50% of its pixels are skin color, but less than 100%,
            # and if the area is bigger than an arbritary number

            # print "in label loop"

            if density > 0.5 and density<1 and props[label-1].area > 50: 
                density_cbcr=np.append(density_cbcr, density)
                area_covered=np.append(area_covered, props[label-1].area/img_size)
                hwratio=np.append(hwratio, props[label-1].major_axis_length/props[label-1].minor_axis_length)
                density_hull=np.append(density_hull, props[label-1].solidity)
                circularity=np.append(circularity, props[label-1].eccentricity)
    
        if len(density_cbcr) == 0:
            density_cbcr = -1*np.ones(1)
        if len(area_covered) == 0:
            area_covered = -1*np.ones(1)
        if len(hwratio) == 0:
            hwratio = -1*np.ones(1)
        if len(density_hull) == 0:
            density_hull = -1*np.ones(1) 
        if len(circularity) == 0:
            circularity = -1*np.ones(1)

        labels_df['percent_skin_abg'][index] = sm.percent_skin(data.ImgReg.skin.abg)
        labels_df['percent_skin_cbcr'][index] = sm.percent_skin(data.ImgReg.skin.cbcr)
        labels_df['percent_skin_cccm'][index] = sm.percent_skin(data.ImgReg.skin.cccm)
        labels_df['percent_skin_equalized_abg'][index] = sm.percent_skin(data.ImgEq.skin.abg)
        labels_df['percent_skin_equalized_cbcr'][index] = sm.percent_skin(data.ImgEq.skin.cbcr)
        labels_df['percent_skin_equalized_cccm'][index] = sm.percent_skin(data.ImgEq.skin.cccm)
        labels_df['label_number'][index] = len(density_cbcr)
        labels_df['area_covered_max'][index] = max(area_covered)
        labels_df['area_covered_mode'][index] = float(stats.mode(area_covered)[0])
        labels_df['density_max'][index] = max(density_cbcr)
        labels_df['density_mode'][index] = float(stats.mode(density_cbcr)[0])
        labels_df['hwratio_max'][index] = max(hwratio)
        labels_df['hwratio_mode'][index] = float(stats.mode(hwratio)[0])
        labels_df['density_hull_max'][index] = max(density_hull)
        labels_df['density_hull_mode'][index] = float(stats.mode(density_hull)[0])
        labels_df['circularity_max'][index] = max(circularity)
        labels_df['circularity_mode'][index] = float(stats.mode(circularity)[0])

    except HTTPError:
        continue

labels_df.to_csv("Features.csv")
