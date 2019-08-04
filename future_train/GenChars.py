
# coding: utf-8

import os
import argparse

import cv2
import numpy as np
import collections

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

from imgaug import augmenters as iaa

FONT_DIR = "fonts/" # contain fonts to gen font templates
#characters extracted from input fonts
TEMPLATE_FOLDER = "fonts/templates/" 
#characters extracted from real license plate
REAL_DATA_FOLDER = "fonts/real_chars/" 
#used to containt augumented characters
CHAR_FOLDER  = "fonts/chars/" 

OUTPUT_SHAPE = (12, 28)

DIGITS = "0123456789"
LETTERS = "ABCDEFGHKLMNPSTUVXYZ"
CHARS=DIGITS+LETTERS


def make_char_ims(font_path):
    width = OUTPUT_SHAPE[0]
    height = OUTPUT_SHAPE[1]
    
    font = ImageFont.truetype(font_path, height+3)
    
    for c in CHARS:
        width = font.getsize(c)[0]
        print(c,width,height)
        im = Image.new("RGBA", (width, height), (0, 0, 0))
        
        draw = ImageDraw.Draw(im)
        draw.text((0, 0), c, (255, 255, 255), font=font)
        im = im.resize(OUTPUT_SHAPE, Image.ANTIALIAS)
        
        yield c, np.array(im)[:, :, 0]


def load_fonts(folder_path):
    font_char_ims = {}
    fonts = [f for f in os.listdir(folder_path) if f.endswith('.ttf')]
    for i, font in enumerate(fonts):
        font_char_ims[i] = dict(make_char_ims(os.path.join(folder_path, font)))
        
    return fonts, font_char_ims

def set_up_augmentation():
    # iaa image augmentation setup
    sometimes = lambda aug: iaa.Sometimes(0.6, aug)
    seq = iaa.Sequential([
        sometimes(iaa.Affine(
            shear=(-5, 5),  # shear by -16 to +16 degrees
            cval=0,  # if mode is constant, use a cval between 0 and 255
        )),
        sometimes(iaa.CoarseDropout(
            p=0.03,
            size_percent=0.3
        )),
        sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1))),
        sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.04))),
    ])
    return seq

def get_augmented_images():
    images = []
    labels = []

    for folder in [TEMPLATE_FOLDER, REAL_DATA_FOLDER]:
        for fileName in os.listdir(folder):
            try:
                # many char image have the same name
                label = fileName.split(".")[0].split("_")[-1]
                img = cv2.imread(folder + fileName, 0)
                
                labels.append(label)
                images.append(img)
            except:
                pass

    return images, labels

def main():
    parser = argparse.ArgumentParser(usage="To gen fonts or gen augumented chars")
    parser.add_argument("--font", type=bool, default=False, help="set '--font True' to gen font")
    parser.add_argument("--gen_char", type=int, default=0, help="--gen_char NUMBER_OF_ROUND_TO_GEN")
    parser.add_argument("--mode_csv", type=bool, default=False, help="set '--mode_csv True' to output to a CSV file")
    args = parser.parse_args()
    print(args)
    
    ### Gen fonts
    if hasattr(args, "font") and args.font:
        print("Start gen font template")
        variation = 1.0
        # Load font
        fonts, font_char_ims = load_fonts(FONT_DIR)
        if not os.path.isdir(TEMPLATE_FOLDER):
            os.makedirs(TEMPLATE_FOLDER)

        for i in font_char_ims:
            for key, char in font_char_ims[i].items():
                font_name = fonts[i].split(".")[0]
                cv2.imwrite(TEMPLATE_FOLDER + font_name + "_" + key + ".png", char)

        print("Font generation completed")

    ### Gen chars
    if hasattr(args, "gen_char") and args.gen_char:
        print("Start gen training images")
        if not os.path.isdir(CHAR_FOLDER):
            os.makedirs(CHAR_FOLDER)
        seq = set_up_augmentation()
        images, labels = get_augmented_images()
    
        num_of_round = args.gen_char
        print('???')
        if hasattr(args,'mode_csv') and args.mode_csv:
            images = np.asarray(images)
            labels = np.asarray(labels, dtype="U")
            n_images=images.shape[0]
            for i in range(num_of_round):
                print("Gen Iteration: ", i)
                aug_images = seq.augment_images(images)
                flatten_aug_imgs = aug_images.reshape(n_images, OUTPUT_SHAPE[0] * OUTPUT_SHAPE[1])
                labeled_imgs = np.insert(np.asarray(flatten_aug_imgs, dtype="U"), 0, labels, axis=1)
                print("Saving augmented chars")

                with open("./fonts/chars.csv", 'ab') as chars:
                    np.savetxt(chars, labeled_imgs, delimiter=",", fmt="%s")
            print("CSV contain unicode, should use pandas to open data")
        else:
            for path in labels:
                if not os.path.isdir('fonts/data/'+path):
                    os.makedirs('fonts/data/'+path)
            for j in range(num_of_round):
                print("Gen Iteration: ", j)
                aug_images = seq.augment_images(images)
                for i,label in enumerate(labels):
                    cv2.imwrite("fonts/data/"+label+'/number_'+str(j)+'.png',aug_images[i])

        print("Character generation completed")
        return
    
    print("Wrong argument. Check with -h")
    
if __name__ == "__main__":
    main()