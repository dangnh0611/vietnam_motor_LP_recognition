import os
import cv2
import matplotlib.pyplot as plt 
import matplotlib.patches as patches
import numpy as np 
import os
import sys
from skimage.filters import threshold_otsu
from skimage.transform import resize
import matplotlib.lines as mlines
import argparse

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path+"/../localization")
from canny import localizeLP


def characterSegmentation(gray_plate):
	if gray_plate is None:
		return 0
	height=gray_plate.shape[0]
	width=gray_plate.shape[1]
	gray_plate1=gray_plate

	thresh_plate=cv2.adaptiveThreshold(gray_plate1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,13,14)
	w=gray_plate1.shape[1]
	h=gray_plate1.shape[0]
	y_ax=[sum(thresh_plate[i,:]) for i in range(h)]

	max_idx=0
	max_val=0
	for i in range(h//3):
		if y_ax[i]>=max_val:
			max_idx=i
			max_val=y_ax[i]
	top_upper_divide=max_idx
	max_val=0
	for i in range(2*h//3,h):
		if y_ax[i]>max_val:
			max_idx=i
			max_val=y_ax[i]
	bottom_down_divide=max_idx
	max_val=0
	for i in range(h//3,2*h//3):
		if y_ax[i]>max_val:
			max_idx=i
			max_val=y_ax[i]
	vertical_divide=max_idx
	up_im=thresh_plate[top_upper_divide:vertical_divide,:]
	down_im=thresh_plate[vertical_divide+1:bottom_down_divide,:]
	x_ax1=[sum(up_im[:,i]) for i in range(w)]
	x_ax2=[sum(down_im[:,i]) for i in range(w)]
	vm=max(x_ax1)
	xl=0
	xr=0

	border_list=[]
	while True:
		xm=x_ax1.index(max(x_ax1))
		for i in range(xm-1,-1,-1):
			if x_ax1[i]<0.8*x_ax1[xm]:
				xl=i
				break
			else:
				x_ax1[i]=0
		for i in range(xm+1,w):
			if x_ax1[i]<0.8*x_ax1[xm]:
				xr=i
				break
			else:
				x_ax1[i]=0
		if x_ax1[xm]<0.86*vm:
			break
		else:
			border_list.append(xl)
			border_list.append(xr)
			x_ax1[xm]=0
		
	fig,ax=plt.subplots(3,2)
	for line in border_list:
		l = mlines.Line2D([line,line], [0,30])
		ax[1,0].add_line(l)
	ax[0,1].plot(y_ax)
	ax[0,0].imshow(thresh_plate,cmap='gray')
	ax[1,0].imshow(up_im,cmap='gray')
	ax[1,1].imshow(down_im,cmap='gray')
	ax[2,0].plot(x_ax1)
	ax[2,1].plot(x_ax2)
	plt.show(block=False)
	plt.pause(5)
	plt.close()


def main():
	fig_size = plt.rcParams["figure.figsize"]
	print("Current size:", fig_size)
	fig_size[0] = 20
	fig_size[1] = 9
	plt.rcParams["figure.figsize"] = fig_size

	parser = argparse.ArgumentParser(usage="Visualize license plate localization using Canny edge detection")
	parser.add_argument("--folder", type=str, default='park', help="Visualize images in the specified folder.")
	parser.add_argument("--path", type=str, default='', help="Visualize image with the specified path.")
	args = parser.parse_args()

	paths=[]
	if hasattr(args,'folder') and args.folder:
		files = os.listdir(args.folder)
		paths=[os.path.join(args.folder,file) for file in files]
	elif hasattr(args,'path') and args.path:
		paths=[args.path]
	else:
		print('Error: Unexpected arguments.')

	for file in paths:
		print(file)
		img=cv2.imread(file)
		plate,contours=localizeLP(img)
		characterSegmentation(plate)

if __name__=="__main__":
		main()

