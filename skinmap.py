import numpy as np

def RGB_SkinMap(image_data, alpha0=0.1276, beta0=0.9498, gamma0=2.77):
    ''' Skin color selection based on Sudeep's blog '''
        
    R = image_data[:,:,0].astype('f')
    G = image_data[:,:,1].astype('f')
    B = image_data[:,:,2].astype('f')
    
    r = R/(R+G+B)
    g = G/(R+G+B)
    b = B/(R+G+B)
    
    alpha = 3*b*r**2/(r+g+b)**3
    beta = (r+g+b)/(3*r) + (r-g)/(r+g+b)
    gamma = (r*b+g**2)/(g*b)

    return ((alpha>alpha0)&(beta<=beta0)&(gamma<=gamma0)).astype('f')
    
    
def CbCr_SkinMap(image_data, y_min=80, cb_min=90, cb_max=125, cr_min=140, cr_max=160):
    ''' Skin Color from Maurizio '''
    
    R = image_data[:,:,0].astype('f')
    G = image_data[:,:,1].astype('f')
    B = image_data[:,:,2].astype('f')

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
    
    R = image_data[:,:,0].astype('f')
    G = image_data[:,:,1].astype('f')
    B = image_data[:,:,2].astype('f')
    
    r = R/(R+G+B)
    g = G/(R+G+B)
    b = B/(R+G+B)
    
    newdata = np.dstack([r,g,b])
    
    I = np.dot(newdata, np.array([0.2989, 0.5870, 0.1402]))
    cI = np.max(newdata[:,:,1:],axis=2)
    e = I-cI

    return np.logical_and(0.026 <= e, e <= .12)
