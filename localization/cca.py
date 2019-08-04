from skimage import measure
from skimage.measure import regionprops
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import sys
import cv2
import argparse

def localizeLP(img):
	""" Localize license plate area by connected component analysis algorithm."""
	# convert to gray
	gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	# get binary image by apply Otsu thresholding
	ret,thresh_img=cv2.threshold(gray_img,0,255,cv2.THRESH_OTSU)
	# this gets all the connected regions and groups them together
	label_image = measure.label(thresh_img)
	# getting the maximum width, height and minimum width and height that a license plate can be
	img_height=gray_img.shape[0]
	img_width=gray_img.shape[1]
	min_height, max_height, min_width, max_width = (0.1*img_height, 0.7*img_height, 0.1*img_width, 0.7*img_width)
	horizontal_min, horizontal_max = (0.1*img_width, 0.9*img_width)
	
	plate_objects_cordinates = []
	plate_like_objects = []
	# regionprops creates a list of properties of all the labelled regions
	for region in regionprops(label_image):
		# the bounding box coordinates
		min_row, min_col, max_row, max_col = region.bbox
		region_height = max_row - min_row
		region_width = max_col - min_col
		scale=region_height/region_width
		# ensuring that the region identified satisfies the condition of a typical license plate
		if region_height >= min_height and region_height <= max_height and region_width >= min_width and region_width <= max_width and min_col>horizontal_min and max_col<horizontal_max and scale<10/7 and scale>0.7:
			plate_objects_cordinates.append((min_row, min_col,max_row, max_col))
	return plate_objects_cordinates

def main():
	parser = argparse.ArgumentParser(usage="Visualize license plate localization using Canny edge detection")
	parser.add_argument("--folder", type=str, default='park', help="Visualize images in the specified folder.")
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
		cors=localizeLP(img)
		fig, (ax1) = plt.subplots(1)
		ax1.imshow(img, cmap="gray")
		for min_row,min_col,max_row,max_col in cors:
			rectBorder = patches.Rectangle((min_col, min_row), max_col-min_col,max_row-min_row, edgecolor="red", linewidth=2, fill=False)
			ax1.add_patch(rectBorder)
		plt.show(block=False)
		plt.pause(1)
		plt.close()

if __name__=='__main__':
	main()
