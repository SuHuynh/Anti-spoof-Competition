import csv
import os
from imutils import paths
import random

imagePaths = []

img_folder_1 = '../../data/normal_camera/train/'
imagePaths_1 = list(paths.list_images(img_folder_1))
imagePaths.extend(imagePaths_1)

img_folder_2 = '../../data/normal_camera_Hai/real_img/train'
imagePaths_2 = list(paths.list_images(img_folder_2))
imagePaths.extend(imagePaths_2)

img_folder_3 = '../../data/normal_camera_Hai/fake_from_monitor/train'
imagePaths_3 = list(paths.list_images(img_folder_3))
imagePaths.extend(imagePaths_3)

img_folder_4 = '../../data/android_smartphone/train'
imagePaths_4 = list(paths.list_images(img_folder_4))
imagePaths.extend(imagePaths_4)

img_folder_5 = '../.../data/android_smartphone/valid'
imagePaths_5 = list(paths.list_images(img_folder_5))
imagePaths.extend(imagePaths_5)

img_folder_6 = '../../data/normal_camera_Hai/fake_lamsonphoto'
imagePaths_6 = list(paths.list_images(img_folder_6))
imagePaths.extend(imagePaths_6)



print('total training images: ', len(imagePaths))

random.shuffle(imagePaths)
num_real = 0
num_fake = 0

with open('data_train_all.csv', 'w') as csvfile:

    writer = csv.writer(csvfile)
    writer.writerow(['img', 'gt_real', 'gt_fake', 'gt_video', 'gt_photo'])

    for img_path in imagePaths:

        # i = random.randint(0,len(imagePaths)-1)
        flag_1 = 'real_img' in img_path

         # write the training section
        if flag_1==True:
            num_real = num_real+1
            writer.writerow([img_path[3:], 1, 0, 0, 0])

        else:
            num_fake = num_fake+1

            if 'printed_photo' in img_path:

                writer.writerow([img_path[3:], 0, 1, 0, 1])
            else:

                writer.writerow([img_path[3:], 0, 1, 1, 0])

print("number of real img for train: ", num_real)
print("number of fake img for train: ", num_fake)

# ############################################################################
imagePaths = []

img_folder_1 = '../../data/normal_camera/valid/'
imagePaths_1 = list(paths.list_images(img_folder_1))
imagePaths.extend(imagePaths_1)

img_folder_2 = '../../data/normal_camera_Hai/real_img/valid'
imagePaths_2 = list(paths.list_images(img_folder_2))
imagePaths.extend(imagePaths_2)

img_folder_3 = '../../data/normal_camera_Hai/fake_from_monitor/valid'
imagePaths_3 = list(paths.list_images(img_folder_3))
imagePaths.extend(imagePaths_3)

print('total training images: ', len(imagePaths))

random.shuffle(imagePaths)
num_real = 0
num_fake = 0

with open('data_valid_normal_cam.csv', 'w') as csvfile:

    writer = csv.writer(csvfile)
    writer.writerow(['img', 'gt_real', 'gt_fake', 'gt_video', 'gt_photo'])

    for img_path in imagePaths:

        # i = random.randint(0,len(imagePaths)-1)
        flag_1 = 'real_img' in img_path

         # write the training section
        if flag_1==True:
            num_real = num_real+1

            cls_gt = 1
            writer.writerow([img_path[3:], 1, 0, 0, 0])

        else:
            num_fake = num_fake+1

            if 'printed_photo' in img_path:
                cls_gt = 0
                writer.writerow([img_path[3:], 0, 1, 1, 0])
            else:
                cls_gt = 0
                writer.writerow([img_path[3:], 0, 1, 0, 1])


print("number of real img for valid: ", num_real)
print("number of fake img for valid: ", num_fake)

###########################################################################

imagePaths = []

img_folder_1 = '../../data/normal_camera_Hai/test_japanese/tachibana_mobile_fake'
imagePaths_1 = list(paths.list_images(img_folder_1))
imagePaths.extend(imagePaths_1)

img_folder_2 = '../../data/normal_camera_Hai/test_japanese/tachibana_paper_fake'
imagePaths_2 = list(paths.list_images(img_folder_2))
imagePaths.extend(imagePaths_2)

img_folder_3 = '../../data/normal_camera_Hai/test_japanese/tachibana_real'
imagePaths_3 = list(paths.list_images(img_folder_3))
imagePaths.extend(imagePaths_3)

print('total training images: ', len(imagePaths))

random.shuffle(imagePaths)
num_real = 0
num_fake = 0

with open('data_valid_tachibana.csv', 'w') as csvfile:

    writer = csv.writer(csvfile)
    writer.writerow(['img', 'gt_real', 'gt_fake', 'gt_video', 'gt_photo'])

    for img_path in imagePaths:

        # i = random.randint(0,len(imagePaths)-1)
        flag_1 = 'real' in img_path

         # write the training section
        if flag_1==True:
            num_real = num_real+1

            cls_gt = 1
            writer.writerow([img_path[3:], 1, 0, 0, 0])

        else:
            num_fake = num_fake+1

            if 'paper' in img_path:
                cls_gt = 0
                writer.writerow([img_path[3:], 0, 1, 1, 0])
            else:
                cls_gt = 0
                writer.writerow([img_path[3:], 0, 1, 0, 1])


print("number of real img for valid: ", num_real)
print("number of fake img for valid: ", num_fake)