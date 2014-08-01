import numpy as np
from math import pi
from skimage import exposure
from scipy import ndimage

def getRGB(image):
    R= image[:,:,0].astype("f")
    G=image[:,:,1].astype("f")
    B=image[:,:,2].astype("f")
    return R,G,B

def deNoise(image):
    
    struct1 = ndimage.generate_binary_structure(2, 2)

    #Dilate image
    img_dilated = ndimage.morphology.binary_dilation(image, iterations=1, structure=struct1)
    #Fill holes in the image
    img_filled = ndimage.morphology.binary_fill_holes(img_dilated)
    #Erode image
    img_eroded = ndimage.morphology.binary_erosion(img_filled, iterations = 2, structure = struct1)
    
    return img_eroded

def TSL_SkinMap(image_data):
    ''' Skin color selection from http://en.wikipedia.org/wiki/TSL_color_space '''
    (R,G,B) = getRGB(image_data)

    r = R/(R+G+B)
    g = G/(R+G+B)
    b = B/(R+G+B)
    
    rp = r - 1/3.
    gp = g - 1/3.
    
    T = np.zeros(r.shape)
    T[gp>0] = 1/(2*pi)*np.arctan2(rp[gp>0],gp[gp>0]) + 1/4
    T[gp<0] = 1/(2*pi)*np.arctan2(rp[gp<0],gp[gp<0]) + 3/4
    
    S = np.sqrt(9/5.*(rp**2 + gp**2))
    
    L = 0.299*R + 0.587*G + 0.114*B
    
    cond1 = np.logical_and(T > .4, T < .6)
    cond2 = np.logical_and(S > .038, S < .25)
    cond3 = L >= 80
    
    return np.logical_and(cond1, np.logical_and(cond2, cond3))
    

def GrayScale(image):
    ''' Convert to grayscale'''

    (R,G,B) = getRGB(image)    
    return 0.2126*R + 0.7152*G + 0.0722*B


def RGB_SkinMap2(image):
    
    (R,G,B) = getRGB(image)

    cond1 = np.logical_and(R > 95, G > 40)
    cond2 = np.logical_and(B > 20, np.abs(R - G))
    cond3 = np.logical_and(R > B, R > G)
    cond4 = (np.max(image, axis=2) - np.min(image, axis=2)) > 15

    cond5 = np.logical_and(cond1, cond2)
    cond6 = np.logical_and(cond3, cond4)
    
    return np.logical_and(cond5, cond6)

def RGB_SkinMap(image_data, alpha0=0.1276, beta0=0.9498, gamma0=2.77):
    ''' Skin color selection based on Sudeep's blog '''
    
    (R,G,B) = getRGB(image_data)

    r = R/(R+G+B)
    g = G/(R+G+B)
    b = B/(R+G+B)

    alpha = 3*b*r**2/(r+g+b)**3
    beta = (r+g+b)/(3*r) + (r-g)/(r+g+b)
    gamma = (r*b+g**2)/(g*b)

    return ((alpha>alpha0)&(beta<=beta0)&(gamma<=gamma0)).astype('f')
    
    
def CbCr_SkinMap(image_data, y_min=80, cb_min=90, cb_max=125, cr_min=140, cr_max=160):
    ''' Skin Color from Maurizio '''

    (R,G,B) = getRGB(image_data)

    y = 0.299*R + 0.587*G + 0.114*B
    cr = R - y + 128
    cb = B - y + 128

    array1 = np.logical_and((cb>cb_min),(cb<cb_max))
    array2 = np.logical_and((cr>cr_min),(cr<cr_max))
    array3 = y > y_min

    return np.logical_and(array1,np.logical_and(array2,array3)).astype('f')    
    
def CCCM_SkinMap(image):
    ''' Skin Color from A Skin Tone Detection Algorithm for an Adaptive Approach to Steganography
        Abbas Cheddad, Joan Condell, Kevin Curran and Paul Mc Kevitt
    '''

    (R,G,B) = getRGB(image)

    r = R/(R+G+B)
    g = G/(R+G+B)
    b = B/(R+G+B)

    newdata = np.dstack([r,g,b])

    I = np.dot(newdata, np.array([0.2989, 0.5870, 0.1402]))
    cI = np.max(newdata[:,:,1:],axis=2)
    e = I-cI

    return np.logical_and(0.026 <= e, e <= .12)


class SkinFilter():
    def __init__(self, image):
        self.cbcr= CbCr_SkinMap(image)
        self.abg= RGB_SkinMap(image)  #alpha beta gamma
        self.cccm= CCCM_SkinMap(image)
        self.tsl= TSL_SkinMap(image)
        self.rgb = RGB_SkinMap2(image)

class Image_Reg():
    def __init__(self,image):
        self.image= image
        self.skin= SkinFilter(self.image)

class Image_HistEq():
    def __init__(self,image):
        self.image= exposure.equalize_hist(image)*255.
        self.skin= SkinFilter(self.image)
        
class Image_HistEqAdapt():
    def __init__(self,image):
        self.image= exposure.equalize_adapthist(image)*255.
        self.skin= SkinFilter(self.image)

class SetUpImage():
    def __init__(self,image):
        self.ImgReg= Image_Reg(image)
        self.ImgEq= Image_HistEq(image)
        self.ImgEqAdapt= Image_HistEqAdapt(image)
        
    def SkinLikelihood(self):
        
        return self.ImgReg.skin.abg + self.ImgEq.skin.abg + self.ImgEqAdapt.skin.abg + \
               self.ImgReg.skin.rgb + self.ImgEq.skin.rgb + self.ImgEqAdapt.skin.rgb + \
               self.ImgReg.skin.cbcr + self.ImgEq.skin.cbcr + self.ImgEqAdapt.skin.cbcr + \
               self.ImgReg.skin.cccm + self.ImgEq.skin.cccm + self.ImgEqAdapt.skin.cccm + \
               self.ImgReg.skin.tsl + self.ImgEq.skin.tsl + self.ImgEqAdapt.skin.tsl
                           
def percent_skin(skin_image):
    return np.shape(np.where(skin_image==1.))[1]/float(skin_image.size)