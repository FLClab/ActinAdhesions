import os
from tqdm import tqdm
import numpy
import tifffile
import matplotlib.pyplot as plt

fgt = 'seg_GT'
fseg = 'seg_actin'
factin = 'actin'
TP_t, FP_t, TP_s, FP_s = [], [], [], []
DC_t = []

def DC(seg,gt,thresh):

	seg = (seg>thresh).astype(int)

	TP = numpy.sum(numpy.logical_and(seg == 1, gt== 1))
	TN = numpy.sum(numpy.logical_and(seg == 0, gt == 0))
	FP = numpy.sum(numpy.logical_and(seg == 1, gt == 0))
	FN = numpy.sum(numpy.logical_and(seg == 0, gt == 1))
	DC = numpy.sum(seg[gt==1])*2.0 / (numpy.sum(seg) + numpy.sum(gt)) 

	return TP/(TP+FN), FP/(FP+TN), DC

score_list = []
thresh_list = numpy.arange(-1,256,1)
for t in tqdm(thresh_list):
	TP_list, FP_list = [], [] 
	DC_list = []

	for file in os.listdir(fgt):
		if '.tif' in file:
			gt = tifffile.imread(os.path.join(fgt, file))
			seg = tifffile.imread(os.path.join(fseg, file.replace('GT','actin')))
			actin = tifffile.imread(os.path.join(factin, file.replace('seg_GT','actin')))

			# Back to [0,255] range
			gt = (gt + 1) / 2 # GT is between 0 and 1
			seg = (seg + 1) / 2 * 255

			TP, FP, dc = DC(seg[actin>-0.9],gt[actin>-0.9],t)

			TP_list.append(TP)
			FP_list.append(FP)
			DC_list.append(dc)

	FP_t.append(numpy.mean(FP_list))
	TP_t.append(numpy.mean(TP_list))
	DC_t.append(numpy.mean(DC_list))

	TP_s.append(numpy.std(TP_list))

	tp_calc = numpy.mean(TP_list)
	fp_calc = numpy.mean(FP_list)

	score_list.append(tp_calc-fp_calc)

best = thresh_list[numpy.argmax(score_list)]
best_dc = thresh_list[numpy.argmax(DC_t)]
print('Best threshold from ROC curve: {} for [0,255], or {} for [-1,1]'.format(best, (best/255)*2-1))
print('Best threshold from DC curve: {} for [0,255], or {} for [-1,1]'.format(best_dc, (best_dc/255)*2-1))

fi2,ax2 = plt.subplots()
plt.title('DC for segmentation of actin')
ax2.set_xlabel('Threshold of the prediction')
plt.plot(thresh_list/255*2-1, DC_t)

fig3,ax3 = plt.subplots()
plt.title('TP-FP for segmentation of actin')
ax3.set_xlabel('Threshold of the prediction')
plt.plot(thresh_list/255*2-1, numpy.array(TP_t)-numpy.array(FP_t))

fig, ax = plt.subplots()
plt.title('seg_actin vs. seg_GT')

TP_t = numpy.array(TP_t)
TP_s = numpy.array(TP_s)

# Compute AUROC
AUROC = 0
for i in range(len(TP_t)-1):
	AUROC += (TP_t[i+1]+TP_t[i])*(FP_t[i]-FP_t[i+1])/2

print('AUROC: ', AUROC)

ax.plot(FP_t, TP_t,'og')
xlim = ax.get_xlim()
ylim = ax.get_ylim()

ax.fill_between(FP_t, TP_t-TP_s, TP_t+TP_s, alpha=0.2, color='g')

ax.plot([0,1], [0,1], '--k')

ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_xlim(xlim)
ax.set_ylim(ylim)

plt.legend()
plt.show()