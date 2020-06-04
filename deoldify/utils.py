import random
import numpy as np
# from skimage.measure import block_reduce
import cv2
from PIL import Image
import os


def mini_norm(x):
    y = x.astype(np.float32)
    y = 1 - y / 255.0
    y -= np.min(y)
    temp = np.max(y)
    if temp>0:
        y /= np.max(y)
    return (255.0 - y * 255.0).astype(np.uint8)


def sensitive(x, s=15.0):
    y = x.astype(np.float32)
    y -= s
    y /= 255.0 - s * 2.0
    y *= 255.0
    return y.clip(0, 255).astype(np.uint8)

def my_resize(img,size,divisible=64):
    h,w = img.shape[0],img.shape[1]
    if h<w:
        target_w = w*size//h
        target_w = target_w//64*64
        return cv2.resize(img,(target_w,size))
    else:
        target_h = h*size//w
        target_h = target_h//64*64
        return cv2.resize(img,(size,target_h))


def generate_user_point(image,img_size=-1, is_random=True):
    h,w,_ = image.shape

    if img_size==-1:
        result = np.zeros((h, w, 4)).astype(np.uint8)
    else:
        result = np.zeros((img_size,img_size,4)).astype(np.uint8)

    if is_random:
        hint_number = int(np.random.normal(20, 10, 1)[0])
        for i in range(hint_number):
            # sample location
            y = int(np.clip(np.random.normal(h/2., h/5.), 0, h-1))
            x = int(np.clip(np.random.normal(w/2., w/5.), 0, w-1))

            # add color point
            color = image[y,x]
            if img_size == -1:
                cv2.circle(result, (x, y), 1, (int(color[0]), int(color[1]), int(color[2]), 255), -1)
            else:
                cv2.circle(result,(int(x*img_size/w),int(y*img_size/h)),1,(int(color[0]),int(color[1]),int(color[2]),255),-1)
    else:
        step = 9
        x_interval = w//step
        y_interval = h//step
        for i in range(1,step):
            for j in range(1,step):
                x = i*x_interval
                y = j*y_interval
                # add color point
                color = image[y,x]
                if img_size == -1:
                    cv2.circle(result, (x, y), 1, (int(color[0]), int(color[1]), int(color[2]), 255), -1)
                else:
                    cv2.circle(result,(int(x*img_size/w),int(y*img_size/h)),1,(int(color[0]),int(color[1]),int(color[2]),255),-1)

    return Image.fromarray(cv2.cvtColor(result,cv2.COLOR_BGRA2RGBA))

class user_point_generator:
    def __init__(self):
        self.hint_number_mu = 20
        self.hint_number_sigma = 10
        self.sample_number = 1000
        self.samples =  np.clip(np.random.normal(self.hint_number_mu, self.hint_number_sigma, self.sample_number), 0, 500)

    def generate(self,image,img_size=-1, is_random=True):

        h,w,_ = image.shape

        if img_size==-1:
            result = np.zeros((h, w, 4)).astype(np.uint8)
        else:
            result = np.zeros((img_size,img_size,4)).astype(np.uint8)

        if is_random:
            hint_number = int(self.samples[random.randint(0,self.sample_number-1)])
            for i in range(hint_number):
                # sample location
                y = int(np.clip(np.random.normal(h/2., h/5.), 0, h-1))
                x = int(np.clip(np.random.normal(w/2., w/5.), 0, w-1))

                # add color point
                color = image[y,x]
                if img_size == -1:
                    cv2.circle(result, (x, y), 1, (int(color[0]), int(color[1]), int(color[2]), 255), -1)
                else:
                    cv2.circle(result,(int(x*img_size/w),int(y*img_size/h)),1,(int(color[0]),int(color[1]),int(color[2]),255),-1)
        else:
            step = 9
            x_interval = w//step
            y_interval = h//step
            for i in range(1,step):
                for j in range(1,step):
                    x = i*x_interval
                    y = j*y_interval
                    # add color point
                    color = image[y,x]
                    if img_size == -1:
                        cv2.circle(result, (x, y), 1, (int(color[0]), int(color[1]), int(color[2]), 255), -1)
                    else:
                        cv2.circle(result,(int(x*img_size/w),int(y*img_size/h)),1,(int(color[0]),int(color[1]),int(color[2]),255),-1)

        return Image.fromarray(cv2.cvtColor(result,cv2.COLOR_BGRA2RGBA))