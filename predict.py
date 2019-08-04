import os
from segmentation import contour
from localization import canny
from sklearn.externals import joblib
from skimage.transform import resize
from skimage.filters import threshold_otsu
import cv2
import time
import sys
import segmentation
import matplotlib.pyplot as plt
import argparse
import warnings
warnings.filterwarnings("ignore")

def predictResult(char_list):
	if char_list is None:
		return 0
	# load the model
	# model_dir_char='model/svc_char.pkl'
	# model_char = joblib.load(model_dir_char)
	# model_dir='model/svc.pkl'
	# model = joblib.load(model_dir)

	model_dir_char='model/letter_model.pkl'
	model_char = joblib.load(model_dir_char)
	model_dir='model/digit_model.pkl'
	model = joblib.load(model_dir)

	classification_result = []
	new_char_list=[]

	for i in range(len(char_list)):
		# converts it to a 1D array
		resized_char_face=resize(char_list[i],(28,12))
		threshold_value = threshold_otsu(resized_char_face)
		ret,binary_char_face=cv2.threshold(resized_char_face,threshold_value,255,cv2.THRESH_BINARY_INV)
		new_char_list.append(binary_char_face)
		char = binary_char_face.reshape(1,-1)
		if i!=2:
			result = model.predict(char)
		else:
			result=model_char.predict(char)
		classification_result.append(result)


	plate_string = ''
	for eachPredict in classification_result:
		plate_string += eachPredict[0]

	print('Final result is:',plate_string)
	return plate_string


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

	for file in paths:
		print(file)
		img=cv2.imread(file)
		plate,contours=canny.localizeLP(img)
		char_list=contour.characterSegmentation(plate)
		if char_list is None:
			continue
		string=predictResult(char_list)
		plt.subplot(221)
		plt.imshow(img)
		plt.subplot(222)
		plt.imshow(plate,cmap='gray')
		plt.title(string)
		for i in range(len(char_list)):
			plt.subplot(2,len(char_list),len(char_list)+i+1)
			plt.imshow(char_list[i],cmap='gray')
		plt.waitforbuttonpress()
				

if __name__=="__main__":
		main()
