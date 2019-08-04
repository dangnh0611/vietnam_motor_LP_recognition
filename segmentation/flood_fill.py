import cv2
import matplotlib.pyplot as plt 
import matplotlib.patches as patches
import numpy as np 
import os
import sys
from skimage.filters import threshold_otsu
from skimage.transform import resize
import argparse

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path+"/../localization")
from canny import localizeLP

def xRect(rect):
	return rect[0]

def characterSegmentation(gray_plate1,visualize=False):
	if gray_plate1 is None:
		return None
	height=gray_plate1.shape[0]
	width=gray_plate1.shape[1]
	gray_plate=gray_plate1[int(0.05*height):int(0.95*height),int(0.03*width):int(0.97*width)]

	try:
		threshold_value = threshold_otsu(gray_plate)
	except ValueError:
		return None

	plate=cv2.adaptiveThreshold(gray_plate,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,13,14)

	height=plate.shape[0]
	width=plate.shape[1]
	mask = np.zeros((height+2, width+2), np.uint8)
	mask[:]=0
	curr_color=1
	rect_list=[]
	for y in range(height):
		for x in range(width):
			point=plate[y,x]
			if point!=255 and point!=0:
				continue
			retval, image, mask, rect=cv2.floodFill(plate,mask,(x,y),curr_color)
			if point==255:
				rect_list.append(rect)
			if curr_color==254:
				curr_color=1
			else:
				curr_color+=1
	min_h,min_w,max_h,max_w=(0.27*height,0.04*width,0.55*height,0.2*width)

	characters=[]
	up=[]
	down=[]
	global_chars=[]
	if visualize:
		min_x,min_y,max_x,max_y=(0.01*width,0.03*height,0.92*width,0.75*height)
		fig,ax=plt.subplots(1,3)
		ax[2].imshow(plate,cmap='gray')
		ax[1].imshow(plate,cmap='gray')
		ax[0].imshow(gray_plate,cmap='gray')
	for rect in rect_list:
		x,y,w,h=rect
		rect_border = patches.Rectangle((x, y), w, h, edgecolor="green",linewidth=2, fill=False)
		if visualize:
			ax[2].add_patch(rect_border)
		if w>min_w and w<max_w and h>min_h and h<max_h:
			rect_border = patches.Rectangle((x, y), w, h, edgecolor="red",linewidth=2, fill=False)
			if visualize:
				ax[2].add_patch(rect_border)
			characters.append(rect)
			if y<3*height/8:
				up.append(rect)
			else:
				down.append(rect)
	print('Number of character detected:',len(up),len(down))
	if len(up)!=4 or (len(down)!=4 and len(down)!=5):
		print("Invalid, can't segment!")
		plt.close()
		return None
	else:
		up= sorted(up, key = xRect)
		down=sorted(down,key=xRect)
		up+=down
		i=0
		char_list=[]
		for x,y,w,h in up:
			char_face=gray_plate[y:y+h,x:x+w]
			char_list.append(char_face)
	plt.show(block=False)
	plt.pause(1)
	plt.close()
	return char_list


def main():
	# plot with full screen
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
		img=cv2.imread(file)
		plate,contours=localizeLP(img)
		char_list=characterSegmentation(plate,visualize=True)

if __name__=="__main__":
		main()

	
