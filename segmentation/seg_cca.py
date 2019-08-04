import numpy as np
from skimage.transform import resize
from skimage import measure
from skimage.measure import regionprops
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
import sys
import os
import argparse
import cv2

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path+"/../localization")
from canny import localizeLP

def characterSegmentation(plate):
	if plate is None:
		return 0
	threshold_value = threshold_otsu(plate)
	plate = plate > threshold_value
	license_plate = np.invert(plate)
	labelled_plate = measure.label(license_plate)

	fig, ax1 = plt.subplots(1)
	ax1.imshow(license_plate, cmap="gray")
	
	character_dimensions = (0.2*license_plate.shape[0], 0.8*license_plate.shape[0], 0.05*license_plate.shape[1], 0.3*license_plate.shape[1])
	min_height, max_height, min_width, max_width = character_dimensions

	characters = []
	column_list = []
	for regions in regionprops(labelled_plate):
		y0, x0, y1, x1 = regions.bbox
		region_height = y1 - y0
		region_width = x1 - x0

		if region_height > min_height and region_height < max_height and region_width > min_width and region_width < max_width:
			roi = license_plate[y0:y1, x0:x1]
			# draw a red bordered rectangle over the character.
			rect_border = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor="red",linewidth=2, fill=False)
			ax1.add_patch(rect_border)
			characters.append(roi)
			column_list.append(x0)

	plt.show(block=False)
	plt.pause(1)
	plt.close()

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
		characterSegmentation(plate)

if __name__=="__main__":
	main()