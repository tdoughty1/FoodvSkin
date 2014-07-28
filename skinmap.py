import numpy as np
from skimage import exposure

def getRGB(image):
    R= image[:,:,0].astype("f")
    G=image[:,:,1].astype("f")
    B=image[:,:,2].astype("f")
    return R,G,B

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
    
    
def CCCM_SkinMap(image_data):
    ''' Skin Color from A Skin Tone Detection Algorithm for an Adaptive Approach to Steganography
        Abbas Cheddad, Joan Condell, Kevin Curran and Paul Mc Kevitt
    '''

    (R,G,B) = getRGB(image_data)

    r = R/(R+G+B)
    g = G/(R+G+B)
    b = B/(R+G+B)

    newdata = np.dstack([r,g,b])

    I = np.dot(newdata, np.array([0.2989, 0.5870, 0.1402]))
    cI = np.max(newdata[:,:,1:],axis=2)
    e = I-cI

    return np.logical_and(0.026 <= e, e <= .12)


class SkinFilter():
    def __init__(self,image):
        self.cbcr= CbCr_SkinMap(image)
        self.abg= RGB_SkinMap(image)  #alpha beta gamma
        self.cccm= CCCM_SkinMap(image)

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


def percent_skin(skin_image):
    return np.shape(np.where(skin_image==1.))[1]/float(skin_image.size)