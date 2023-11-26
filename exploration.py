import os
from PIL import Image
import json
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision
import numpy as np
import torch
import argparse
# import segmentation_utils
# import cv2
from PIL import Image

def load_data(path):
    
    cityscapes_train_cities = os.listdir(path)

    for city in cityscapes_train_cities:
        images = os.listdir(f"{path}/{city}")

        for image in images:
            path_to_image = f"{path}/{city}/{image}"
            
            # if image == "strasbourg_000000_019355_gtFine_color.png":
            #     print("COLORS")
            #     img = Image.open(path_to_image)
            #     img = np.array(img, dtype=np.uint8)
            #     print(img[256*3][1024])
            #     # for row in img:
            #     #     for col in row:
            #     #         if col[0] == 0:
            #     #             break
            #     #         print(col)
            #     #     break
            #     # print(img.shape)
            #     # print(img)
            #     # img.save(f"./{image}")
            # # if image == "strasbourg_000000_019355_gtFine_instanceIds.png":
            # #     img = Image.open(path_to_image)
            # #     img.save(f"./{image}")
            # if image == "strasbourg_000000_019355_gtFine_labelIds.png":
            #     print("LABELS")
            #     img = Image.open(path_to_image)
            #     img = np.array(img, dtype=np.uint8)
            #     print(img[256*3][1024])

            # if image == "aachen_000039_000019_leftImg8bit.png":
            #     img = Image.open(path_to_image)
            #     img.save(f"./input/{image}")
            if image == "aachen_000039_000019_gtFine_color.png":
                img = Image.open(path_to_image)
                img = np.array(img, dtype=np.uint8)
                print("COLOR", img[150][1857])
                # img.save(f"./input/{image}")
            if image == "aachen_000039_000019_gtFine_labelIds.png":
                print("LABELS")
                img = Image.open(path_to_image)
                img = np.array(img, dtype=np.uint8)
                print("ID", img[150][1857])
                for j, row in enumerate(img):
                    if 20 in row:
                        # print(row[10])
                        a = [i for i in range(len(row)) if row[i]==20]
                        print(j, a)
                        break
            
            # status = False
            # if path_to_image[-4:] == ".png":
            #     PILimage = Image.open(path_to_image)
            #     img = np.array(PILimage, dtype=np.uint8)
            #     if path_to_image[-19:] == "gtFine_labelIds.png":
            #         for row in img:
            #             if 7 in row:
            #                 PILimage.save(f"./input/{image}")
            #                 status = True
            #                 break
            # if status:
            #     break
        #     break
        # break

"""
def test_model(dataset_path, model):
    
    # NUMBER OF TEST IMAGES: 1525
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval().to(device)

    raw_test_data_folder = "leftImg8bit_trainvaltest/leftImg8bit/test/"
    cityscapes_raw_test = f"{dataset_path}{raw_test_data_folder}"

    test_cities = os.listdir(cityscapes_raw_test)

    for city in test_cities:
        images = os.listdir(f"{cityscapes_raw_test}{city}")
        print(len(images))
        for i, image in enumerate(images):
            print("TEST IMAGE:", i)
            path_to_image = f"{cityscapes_raw_test}{city}/{image}"
            img = Image.open(path_to_image)

            outputs = segmentation_utils.get_segment_labels(img, model, device)
            # print(outputs['out'])
            outputs = outputs['out']
            print(outputs.shape)
            # segmented_image = segmentation_utils.draw_segmentation_map(outputs)
            # final_image = segmentation_utils.image_overlay(img, segmented_image)

            # img2 = Image.fromarray(segmented_image)
            # img2.save(f"./outputs/baseline/test/{image}")
            break
        break
"""

if __name__ == "__main__":

    cityscapes_folder_path = "../../../../projects/vc/data/ad/open/Cityscapes/"
    gt_folder = "gtFine_trainvaltest/gtFine/train"
    raw_data_folder = "leftImg8bit_trainvaltest/leftImg8bit/train"

    test_dataset_path = "leftImg8bit_trainvaltest/leftImg8bit/test"

    load_data(f"{cityscapes_folder_path}{gt_folder}")

    # # // BASELINE // ================
    # model = torchvision.models.segmentation.fcn_resnet50(pretrained=True)
    # test_model(cityscapes_folder_path, model)






