import cv2
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
import os
import argparse

def localizeLP(img):
	""" Localize image using Cascade classifier with Haar-like features."""
	# preprocess
	gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	# using cascade classifier to detect license plate
	# the trained classifier is not good, retrain to get better results.
	# pre-trained classifier at: https://github.com/openalpr/openalpr/tree/master/runtime_data/region
	cascade = cv2.CascadeClassifier("localization/cascade_model.xml")
	rects = cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30),flags=cv2.CASCADE_SCALE_IMAGE)
	if len(rects)==0:
		return None,None

	rect_list=[(rect[0],rect[1],rect[2],rect[3]) for rect in rects]
	img_height=gray_img.shape[0]
	img_width=gray_img.shape[1]
	min_height, max_height, min_width, max_width = (0.1*img_height, 0.7*img_height, 0.1*img_width, 0.7*img_width)
	horizontal_min, horizontal_max = (0.1*img_width, 0.9*img_width)
	for obj_rect in rect_list:
		min_col, min_row, w, h=obj_rect
		max_col = min_col+w
		max_row=min_row+h
		if h >= min_height and h <= max_height and w >= min_width and w <= max_width and min_col>horizontal_min and max_col<horizontal_max:
			plate=gray_img[min_row:max_row,min_col:max_col]
			return plate,(min_row,min_col,max_row,max_col)


def main():
	parser = argparse.ArgumentParser(usage="Visualize license plate localization using Canny edge detection")
	parser.add_argument("--folder", type=str, default='dataset', help="Visualize images in the specified folder.")
	parser.add_argument("--path", type=str, default='', help="Visualize image with the specified path.")
	args = parser.parse_args()

	paths=[]
	if hasattr(args,'path') and args.path:
		paths=[args.path]
	elif hasattr(args,'folder') and args.folder:
		files = os.listdir(args.folder)
		paths=[os.path.join(args.folder,file) for file in files]
	else:
		print('Error: Unexpected arguments.')

	for filename in paths:
		print(filename)
		img=cv2.imread(filename)
		plate,plate_rect=localizeLP(img)
		fig,(ax1,ax2)=plt.subplots(1,2)
		ax1.imshow(img)
		if plate is not None:
			ax2.imshow(plate,cmap='gray')
			min_row,min_col,max_row,max_col=plate_rect
			rectBorder = patches.Rectangle((min_col, min_row), max_col-min_col, max_row-min_row, edgecolor="red", linewidth=2, fill=False)
			ax1.add_patch(rectBorder)
		plt.show(block=False)
		plt.pause(1)
		plt.close()

if __name__=='__main__':
	main()
