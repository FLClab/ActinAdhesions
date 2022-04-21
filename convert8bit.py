import tifffile
import numpy
import os 

folder = 'seg_actin'

if '8bit' not in os.listdir(folder):
	os.mkdir(os.path.join(folder,'8bit'))

for file in os.listdir(folder):
	if '.tif' in file:
		im = tifffile.imread(os.path.join(folder, file))
		im = (im > -0.3725).astype('uint8')*255#(im + 1) / 2 * 255
		#im = im.astype('uint8')
		tifffile.imsave(os.path.join(folder, '8bit', file), im)