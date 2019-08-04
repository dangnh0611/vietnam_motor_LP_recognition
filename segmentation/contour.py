import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.filters import threshold_otsu
import numpy as np
import sys
import os
import argparse
from operator import itemgetter, attrgetter

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path+"/../localization")
from canny import localizeLP

def characterSegmentation(gray_plate,visualize=False):
	""" Return list of cropped boudding box of each character."""

	if gray_plate is None:
		return None
	# adaptive threshold to convert to binary image
	thre=cv2.adaptiveThreshold(gray_plate,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,13,14)
	# finding contours
	thre,contours,hierachy = cv2.findContours(thre,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
	width=gray_plate.shape[0]
	height=gray_plate.shape[1]
	min_h,min_w,max_h,max_w=(0.2*height,0.04*width,0.6*height,0.3*width)

	#list of characters in each line of license plate
	line1=[]
	line2=[]
	# list all characters
	rect_list=[]

	for c in contours:
		rect = cv2.boundingRect(c)
		(x,y,w,h)=rect
		if w<max_w and w>min_w and h>min_h and h<max_h and w/h>0.15 and w/h<0.6:
			if y<3*height/8:
				line1.append(rect)
			else:
				line2.append(rect)
	if len(line1)!=4 or (len(line2)!=4 and len(line2)!=5):
		print('	Can\'t detect.')
		return None

	line1.sort(key=itemgetter(0))
	line2.sort(key=itemgetter(0))
	rect_list=line1+line2
	print('	Detected',len(rect_list),'characters.')
	char_list=[gray_plate[y:y+h,x:x+w] for (x,y,w,h) in rect_list]

	if visualize:
		fig,(ax1)=plt.subplots(1,1)
		ax1.imshow(thre,cmap='gray')

		for c in contours:
			rect = cv2.boundingRect(c)
			(x,y,w,h)=rect
			if w<max_w and w>min_w and h>min_h and h<max_h and w/h>0.15 and w/h<0.6:
				rect_border = patches.Rectangle((x, y), w, h, edgecolor="red",linewidth=2, fill=False)
				ax1.add_patch(rect_border)
			else:
				rect_border = patches.Rectangle((x, y), w, h, edgecolor="green",linewidth=2, fill=False)
				ax1.add_patch(rect_border)

		plt.show(block=False)
		plt.pause(2)
		plt.close()

	return char_list


def main():
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
		characterSegmentation(plate,visualize=True)

if __name__=="__main__":
	main()