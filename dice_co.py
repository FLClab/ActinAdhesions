import matplotlib.pyplot as plt
import numpy as np
import tifffile
from PIL import Image
import skimage.filters
import os


epoch = ['2000', '2500', '3000', '3500', '4000', '4100', '4200', '4300', '4400', '4500', '4600', '4700', '4800', '4900', '5000', '5100', '5200', '5300', '5400','5500', '5600', '5700', '5800', '5900', '6000']
# segmentation
image_number = ['2', '5', '11', '17', '18']
dice_epoch = []


def average (array, len):
    sum = 0
    for i in range(len):
        sum += array[i]
    return sum/len


for i in epoch:
    folder = './results/adhesion_full_dataset_new/valid_{}/images/'.format(i)
    dice_images = []
    for j in image_number:
        seg_filename = 'chok1_63x_405-actin_488-pax {}_seg_fakeactin.tif'.format(j)
        GT_filename = 'chok1_63x_405-actin_488-pax {}_seg_actin.tif'.format(j)

        seg = tifffile.imread(os.path.join(folder, seg_filename))
        seg_image = np.array(seg)
            #convert to binary mask
        binary_mask = seg_image > -0.7412
    

        binary_mask = binary_mask.astype('int')

            # ground truth
        GT_image = tifffile.imread(os.path.join(folder, GT_filename))
        GT = np.array(GT_image)
        binary_GT = GT > -0.7412
        binary_GT = binary_GT.astype('int')
        dice = np.sum(binary_mask[binary_GT==1])*2.0 / (np.sum(binary_mask) + np.sum(binary_GT))
        dice_images.append(dice)
    dice_epoch.append(average(dice_images, len(dice_images)))

print (dice_epoch)

plt.plot(epoch, dice_epoch)
plt.xlabel('epoch_count')
plt.ylabel('dice_coefficient')
plt.title('Pick the best epoch')

plt.show()        

plt.savefig('dice.png')