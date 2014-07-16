'''Dr. Das' code for the Gomez & Morales (2002) solution
see: http://datamusing.info/blog/2014/07/06/detecting-people-in-photographs-using-skin-tone/
'''
import pylab
import matplotlib.image as mpimg
# read in the image
values = mpimg.imread("images/blog_img2.png")
# separate out r, g, b channels
r = values[:,:,0].astype('f')
g = values[:,:,1].astype('f')
b = values[:,:,2].astype('f')
# generate the three quantities 
alpha = 3*b*r**2/(r+b+g)**3
beta =  (r+g+b)/(3*r) + (r-g)/(r+g+b)
gamma = (r*b+g**2)/(g*b)
# finally we apply the rules:
pylab.imshow((alpha>0.1276)&(beta<=0.9498)&(gamma<=2.7775),cmap='gray')
pylab.show()
