import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
import glob
import matplotlib.patches as patches
from pathlib import Path
import pathlib
import os

def try_argument():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')

    args = parser.parse_args()

    train_kwargs = {'batch_size': args.batch_size}

    def output(**value):
        return value

    print(output(**train_kwargs))
# try_argument()

def check_cuda_available():
    import torch
    print(torch.cuda.is_available())

    import os
    print(os.environ.get('CUDA_PATH'))
# check_cuda_available()

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:] # mask.shape = (3, 1200, 1800), mask.shape[-2:] = (1200, 1800)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1) # shape (1200, 1800, 1) x shape (1, 1, 4) = shape (1200, 1800, 4)
    ax.imshow(mask_image) # plt.gca().imshow(mask_image) plots the 1s and 0s, allowing visualization

def show_box(box, image):
    x_start, y_start = box[0], box[1]
    x_end, y_end = box[2], box[3]
    return cv2.rectangle(image, (x_start, y_start), (x_end, y_end), color=(0,255,0), thickness=2)

# h, w = mask.shape[-2:] <-- what does [-2:] do?
def what_this_array_manipulation():
    arr = np.array([[1,2,3], [4,5,6], [7,8,9]])
    print(arr.shape) # (3, 3)
    print(arr)
    # [[1 2 3]
    #  [4 5 6]
    #  [7 8 9]]
    arr= arr[-2:] # remains the last 2 dimension
    print(arr.shape) # (2, 3)
    print(arr)
    # [[4 5 6]
    #  [7 8 9]]
    arr= arr[-1:] # remains the last 1 dimension
    print(arr.shape) # (2, 3)
    print(arr)
# what_this_array_manipulation()

def array_multiplication_shape():
    color = np.array([1, 2, 3, 0.6])
    print('np.array([1, 2, 3, 0.6]) shape: ', color.shape)
    print(color)
    # reshape an array into a 3-dimensional array where the first and second dimensions have a length of 1 each, 
    # and the third dimension is determined automatically to fit the total number of elements in the original array.
    color = color.reshape(1, 1, -1)
    print('color.reshape(1, 1, -1) shape: ', color.shape) # (1, 1, 4)
    print(color, '\n')
    arr = np.array([[1,2,3], [4,5,6], [7,8,9]])
    h, w = arr.shape[-2:]
    mask_image = arr.reshape(h, w, 1)
    print('arr.reshape(h, w, 1) shape: ',mask_image.shape) # (3, 3, 1)
    print(mask_image , '\n')
    mask_image = arr.reshape(h, w, 1) * color.reshape(1, 1, -1)
    print('arr.reshape(h, w, 1) * color.reshape(1, 1, -1) shape: ', mask_image.shape) # (3, 3, 4)
    print(mask_image)
# array_multiplication_shape()

def what_is_plt_gca():
    # create a numpy array of zeros with dimension of 500 x 500, 
    # then makes the middle 100 rows to  be ones
    array_shape = (500, 500)
    zeros_array = np.zeros(array_shape)
    middle_start = (array_shape[0] - 100) // 2
    middle_end = middle_start + 100
    zeros_array[middle_start:middle_end, :] = 1
    plt.gca().imshow(zeros_array)
    # plt.show()
    ones_indices = np.argwhere(zeros_array == 1)

    # Calculate bounding box coordinates
    min_x = ones_indices[:, 1].min()
    max_x = ones_indices[:, 1].max()
    min_y = ones_indices[:, 0].min()
    max_y = ones_indices[:, 0].max()

    # Create a rectangle patch for the bounding box
    bbox_width = max_x - min_x + 1
    bbox_height = max_y - min_y + 1
    bbox = patches.Rectangle((min_x - 0.5, min_y - 0.5), bbox_width, bbox_height, linewidth=2, edgecolor='r', facecolor='none')

    # Add the bounding box to the plot
    plt.gca().add_patch(bbox)

    plt.show()
# what_is_plt_gca()

def try_selectROI():
    # image = cv2.imread('images/truck.jpg')
    images = [cv2.imread(image) for image in glob.glob("images/bmx-trees/*.jpg")]
    image = images[0]
    cv2.namedWindow("Get_mask", cv2.WINDOW_NORMAL)
    x, y, w, h = cv2.selectROI("Get_mask", image, showCrosshair=False, fromCenter=False)
    input_box = np.array([x, y, x+w, y+h])
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image * np.zeros(image.shape)
    image = show_box(input_box, image)
    cv2.imshow('', image)
    cv2.imshow('', images[60])
    cv2.waitKey(0)
# try_selectROI()

def try_mask():
    images = [cv2.imread(image) for image in glob.glob("images/bmx-trees/*.jpg")]
    image = images[0]
    # create a numpy array of zeros with dimension of 500 x 500, 
    # then makes the middle 100 rows to  be ones
    array_shape = (1080, 1920)
    mask = np.zeros(array_shape, dtype=bool)
    middle_start = (array_shape[0] - 100) // 2
    middle_end = middle_start + 200
    mask[middle_start:middle_end, :] = True

    mask = (mask * 255.).astype(np.uint8)

    # mask = ~mask
    # mask = mask.astype(np.uint8) * 1
    # h, w = mask.shape[-2:]
    # mask = mask.reshape(h, w, 1)
    # cv2.imshow('', mask)
    # cv2.waitKey(0)

    # color = np.array([255/255, 255/255, 255/255, 1])
    # print(color.dtype)
    # h, w = mask.shape[-2:]
    # mask = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    # print(mask.dtype)

    cv2.imshow('', mask)
    cv2.waitKey(0)
    cv2.imwrite('C:/Users/Kainian/Desktop/WorkSpace/IM_Ghost_Project/images/annotation/mask.jpg', mask)
    # mask_image = image * mask
    # print(mask_image.shape)


    # cv2.imshow('', mask_image)
    # cv2.waitKey(0)
# try_mask()

def draw_rectangle_from_points():
    images = [cv2.imread(image) for image in glob.glob("images/bmx-trees/*.jpg")]
    image = images[-1]

    location = np.array([559.13275, 703.2408, 573.11835, 370.38312, 771.6838, 378.72626, 757.6982, 711.5839])
    # temp_location = np.intp(location).reshape((-1, 1, 2))
    temp_location = np.reshape(np.intp(location), (4, -1))
    max_point = np.max(temp_location, axis=0)
    min_point = np.min(temp_location, axis=0)
    print(max_point,min_point)
    cv2.rectangle(image, min_point, max_point, color=(0,255,0), thickness=2)

    cv2.imshow('', image)
    cv2.waitKey(0)
# draw_rectangle_from_points()

def create_video_from_images():
    images = [cv2.imread(image) for image in glob.glob("Human6/*.jpg")]

    h, w = images[0].shape[:2]
    size = (w, h)

    name = 'human.mp4'
    out = cv2.VideoWriter(name, cv2.VideoWriter_fourcc(*'mp4v'), 30, size)
    for i in range(len(images)):
        out.write(images[i])
    out.release()
# create_video_from_images()

def create_images_from_video():
    cam = cv2.VideoCapture("video.mp4")
    currentframe = 0
    while(True):
        ret,frame = cam.read()
        if ret:
            cv2.imwrite('video_frame/{:05d}.jpg'.format(currentframe), frame)
            currentframe += 1
        else:
            break
    cam.release()
# create_images_from_video()

def resize_image():
    images = [cv2.imread(image) for image in glob.glob("soccer/*.jpg")]

    for i, image in enumerate(images):
        dim = (854, 480)
        resized_image = cv2.resize(image, dim)
        cv2.imwrite('./images/soccer/{:05d}.jpg'.format(i), resized_image)
# resize_image()

def image_to_gif():
    frames = [Image.open(image) for image in glob.glob('C:/Users/Kainian/Desktop/WorkSpace/Real_Time_Video_Inpainting_PoC/tennis/*.jpg')]
    frame_one = frames[0]
    frame_one.save('C:/Users/Kainian/Desktop/WorkSpace/Real_Time_Video_Inpainting_PoC/gif/tennis_mask_image.gif', format='GIF', append_images=frames, save_all=True, loop=0)
# image_to_gif()

def gif_to_video():
    gif = cv2.VideoCapture('4.gif')

    # store the images read from gif into a list
    images = []

    while(True):
        ret,frame = gif.read()
        if ret:
            images.append(frame)
        else:
            break
    gif.release()

    h, w = images[0].shape[:2]
    size = (w, h)

    name = 'human_fall_4.mp4'
    out = cv2.VideoWriter(name, cv2.VideoWriter_fourcc(*'mp4v'), 30, size)
    for i in range(len(images)):
        out.write(images[i])
    out.release()
# gif_to_video()

def video_to_gif():
    images = []
    frames = []

    video_path = './video/video_4.mp4'

    video = cv2.VideoCapture(video_path)
    while(True):
        ret,frame = video.read()
        if ret:
            images.append(frame)
        else:
            break
    video.release()

    for image in images:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        image = Image.fromarray(image)
        frames.append(image)
    
    frame_one = frames[0]
    frame_one.save('C:/Users/Kainian/Desktop/WorkSpace/new_project/gif/fall/not_fall_train_4.gif', 
                   format='GIF', append_images=frames, save_all=True, loop=0)
# video_to_gif()

def input_video_output_resize_video():
    def func(image):
        h, w, _ = image.shape
        want_width = 1920
        pad_width = int((want_width - w)/2)

        want_height = 1280
        pad_height = int((want_height - h)/2)

        top = pad_height
        bottom = pad_height
        left = pad_width
        right = pad_width

        # Define the padding color (you can use a tuple with BGR values)
        padding_color = (0, 0, 0)  # Black

        # Add padding to the image
        padded_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color)

        return padded_image

    video_paths = list(pathlib.Path('./dataset_val/').glob("*/*.*"))

    count = 1

    for video_path in video_paths:
        if count > 4:
            count = 1

        images = []
        resized_image = []

        video_path = str(video_path).split("/")[0]

        video = cv2.VideoCapture(video_path)
        while(True):
            ret,frame = video.read()
            if ret:
                images.append(frame)
            else:
                break
        video.release()
        
        # h, w, _ = images[0].shape
        # size = (w, h)

        for image in images:
            # image = func(image)
            dim = (960, 720)
            image = cv2.resize(image, dim)
            resized_image.append(image)
        h, w, _ = resized_image[0].shape
        size = (w, h)

        save_video_name = str(os.path.dirname(video_path)) + '/video_' + str(count) + '.mp4'

        out = cv2.VideoWriter(save_video_name, cv2.VideoWriter_fourcc(*'mp4v'), 30, size)
        for i in range(len(resized_image)):
            out.write(resized_image[i])
        # for i in range(len(images)):
        #     out.write(images[i])
        out.release()

        count += 1
# input_video_output_resize_video()



def pad_img(image):
        pad_width = 5
        pad_height = 5

        top = pad_height
        bottom = pad_height
        left = pad_width
        right = pad_width

        padding_color = (255, 255, 255)  # Black

        # Add padding to the image
        padded_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color)
        return padded_image

def read_gif(path):
    images = []
    gif = cv2.VideoCapture(path)
    while(True):
        ret,frame = gif.read()
        if ret:
            images.append(frame)
        else:
            break
    gif.release()
    return images

def concat_train(img_lst, size):
    # num_frames = len(min(img_lst, key=len))
    num_frames = 58
    half_img_lst = int((len(img_lst))/2) # 4

    frames = []

    for i in range(num_frames):
        img_sub = []
        for images in img_lst:
            img_sub.append(pad_img(images[i]))

        concat_v1 = cv2.vconcat(img_sub[ :half_img_lst])
        concat_v2 = cv2.vconcat(img_sub[half_img_lst: ])
        concat_h = cv2.hconcat([concat_v1, concat_v2])
        concat_h = cv2.resize(concat_h, size)
    
        # concat_h = cv2.cvtColor(concat_h, cv2.COLOR_BGR2RGB) 
        
        # # put text
        # concat_h = cv2.putText(concat_h, 'fall', (100,50), fontFace=cv2.FONT_HERSHEY_SIMPLEX,  
        #            fontScale=1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        # concat_h = cv2.putText(concat_h, 'not fall', (426,50), fontFace=cv2.FONT_HERSHEY_SIMPLEX,  
        #            fontScale=1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        
        # # conver for using in PIL
        # concat_h = Image.fromarray(concat_h)

        frames.append(concat_h)
    return frames

def concat_val(img_lst, size):
    num_frames = 58
    frames = []

    for i in range(num_frames):
        concat_v = cv2.vconcat([pad_img(img_lst[0][i]), pad_img(img_lst[1][i])])
        concat_v = cv2.resize(concat_v, size)
        frames.append(concat_v)
    return frames


gif_paths = list(pathlib.Path('gif/fall').glob("*/*.*"))

img_lst = dict()

for gif_path in gif_paths:
    folder_name = str(os.path.basename(os.path.dirname(gif_path)))

    if folder_name in img_lst:
        img_lst[folder_name].append(read_gif(str(gif_path)))
    else:
        img_lst[folder_name] = [read_gif(str(gif_path))]

concate_train = concat_train(img_lst['train'], (500, 800))
concate_val = concat_val(img_lst['val'], (300, 800))

center_img = cv2.imread('video/img.jpg')
center_img = cv2.resize(center_img, (800,800))

frames = []
for i in range(len(concate_train)):
    concat_h = cv2.hconcat([concate_train[i], center_img, concate_val[i]])
    concat_h = cv2.cvtColor(concat_h, cv2.COLOR_BGR2RGB)
    concat_h = Image.fromarray(concat_h)
    frames.append(concat_h)

frame_one = frames[0]
frame_one.save('C:/Users/Kainian/Desktop/WorkSpace/new_project/gif/gif.gif', 
                format='GIF', append_images=frames, save_all=True, loop=0)

