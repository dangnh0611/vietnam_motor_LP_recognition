import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import sys
import os
import argparse
def localizeLP(img):
	""" Function to localize the license plate on a image.
		Return the license plate in bird-eyes view cropped from the image."""
	#Preprocessing image
	gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	# remove noise by bilateral filter
	noise_removal_img = cv2.bilateralFilter(gray_img,9,75,75)
	# histogram equalization
	equal_histogram_img = cv2.equalizeHist(noise_removal_img)

	kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
	morph_img = cv2.morphologyEx(equal_histogram_img,cv2.MORPH_OPEN,kernel,iterations=20)
	sub_morp_image = cv2.subtract(equal_histogram_img,morph_img)

	# threshold Otsu to get binary image
	ret,thresh_image = cv2.threshold(sub_morp_image,0,255,cv2.THRESH_OTSU)
	# canny edge detection
	canny_image = cv2.Canny(thresh_image,250,255)
	# dilate to remove noise
	kernel = np.ones((3,3), np.uint8)
	dilated_image = cv2.dilate(canny_image,kernel,iterations=1)

	#finding contours
	dilated_image,contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	# listing the top 10 contours with largest areas
	large_contours= sorted(contours, key = cv2.contourArea, reverse = True)[:10]

	img_height=gray_img.shape[0]
	img_width=gray_img.shape[1]
	min_height, max_height, min_width, max_width = (0.1*img_height, 0.7*img_height, 0.1*img_width, 0.7*img_width)
	horizontal_min, horizontal_max = (0.1*img_width, 0.9*img_width)
	curr_min_area_error=99999999
	plate=None
	# iterate through the contours list to find the ones with sastified any conditions
	license_plate_rect=None
	for contour in large_contours:
		# find the contour that can be approx by a 4-edges polyDP
		peri = cv2.arcLength(contour, True)
		approx = cv2.approxPolyDP(contour, 0.06 * peri, True) 
		if len(approx)!=4: continue
		rect=np.array(order_points(approx[:,0]),dtype='float32')
		tl, tr, br, bl = rect

		# find the width and height of the approximated rectangle from this current contour
		widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
		widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
		region_width = max(int(widthA), int(widthB))
	
		heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
		heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
		region_height = max(int(heightA), int(heightB))

		if region_height==0 or region_width==0:
			continue
		scale=region_height/region_width

		# check if it sastified some condition about coordinate, width, height, aspect ratio
		if region_height >= min_height and region_height <= max_height and region_width >= min_width and region_width <= max_width and tl[0]>horizontal_min and tr[0]<horizontal_max and scale<10/7 and scale>0.7: 
			# only look for one with the approximated rectange is better
			# in this case, find the one which region area is nearest to the original contour's
			area_error=abs(cv2.contourArea(approx)-region_height*region_width)
			if area_error<curr_min_area_error:
				curr_min_area_error=area_error
				license_plate_rect=(rect,region_height,region_width)
		
	if license_plate_rect is None:
		return None,contours
	# Now, we get the most suitable one
	rect,region_height,region_width=license_plate_rect
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[region_width - 1, 0],
		[region_width - 1, region_height - 1],
		[0, region_height - 1]], dtype = "float32")
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	plate = cv2.warpPerspective(img, M, (region_width, region_height))
		
	ret_plate=cv2.cvtColor(plate,cv2.COLOR_BGR2GRAY)
	return ret_plate,contours


def order_points(pts):
	""" Sort the 4 points in argument to the suitable order."""
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")

	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
 
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
 
	# return the ordered coordinates
	return rect

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
		plate,contours=localizeLP(img)
		if plate is None:
			print('	Can\'t detect!')
			continue
		# visualize localization results
		fig, (ax1,ax2) = plt.subplots(1,2)
		cv2.drawContours(img, contours, -1, (0,255,0), 1)
		ax1.imshow(img)
		ax2.imshow(plate,cmap='gray')
		plt.show(block=False)
		plt.pause(1)
		plt.close()

if __name__=='__main__':
	main()
