import tifffile
import numpy
import os 

folder_a = 'seg_actin'
folder_b = 'seg_GT'

if 'overlay' not in os.listdir(os.getcwd()):
	os.mkdir('overlay')

for file in os.listdir(folder_a):
	if '.tif' in file:
		im_a = (tifffile.imread(os.path.join(folder_a, file)) > -0.3725).astype('uint8') * 255
		im_b = (tifffile.imread(os.path.join(folder_b, file.replace('actin.tif','GT.tif'))) + 1)/2*255

		overlay = numpy.zeros((1024, 1024, 3)) # RGB

		overlay[:,:,0] = im_a # Red
		overlay[:,:,1] = im_b # Green
		overlay[:,:,2] = im_a + im_b # Blue

		tifffile.imsave(os.path.join('overlay', file), overlay)
