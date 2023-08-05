#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：U-2-Net 
@File    ：remove_bg.py
@IDE     ：PyCharm 
@Author  ：xuzhif
@Date    ：8/5/2023 5:22 PM 
"""
import numpy as np
from PIL import Image
import os
from rembg import remove, new_session


def apply_background_color(img, color):
    r, g, b, a = color
    print(r,g,b,a)
    colored_image = Image.new("RGBA", img.size, (r, g, b, a))
    colored_image.paste(img, mask=img)

    return colored_image


def crop(img_file, mask_file):
    # name, *_ = img_file.split(".")
    img_array = np.array(Image.open(img_file))
    mask = np.array(Image.open(mask_file))
    # print(mask)

    # 通过将原图和mask图片归一化值相乘，把背景转成黑色
    # 从mask中随便找一个通道，cat到RGB后面，最后转成RGBA
    # res = np.concatenate((img_array * (mask/255), mask[:, :, [0]]), -1)
    # print(res.shape)
    # print(mask)
    res = np.concatenate((img_array, mask[:, :, [0]]), -1)
    # print(res)
    img = Image.fromarray(res.astype('uint8'), mode='RGBA')
    img = apply_background_color(img, (255, 255, 255, 255))
    # img = img.convert(mode="RGB")
    # print(img)
    # img.show()
    return img


def main():
    model = "u2net"
    # model = "u2netp"

    img_root = os.path.join(os.getcwd(), "test_data/test_images")
    mask_root = os.path.join(os.getcwd(), "test_data/{}_results".format(model))
    crop_root = os.path.join(os.getcwd(), "test_data/{}_crops".format(model))
    if not os.path.exists(crop_root):
        os.makedirs(crop_root, mode=0o775, exist_ok=True)

    for img_file in os.listdir(img_root):
        print("crop image {}".format(img_file))
        name, *_ = img_file.split(".")
        res = crop(
            os.path.join(img_root, img_file),
            os.path.join(mask_root, name + ".png")
        )
        res.save(os.path.join(crop_root, name + "_crop.png"))

        remove_use_rembg(os.path.join(img_root, img_file), os.path.join(crop_root, name + "_rembg.png"))
        # exit()


def remove_use_rembg(input_path, output_path):
    input = Image.open(input_path)
    # sam
    session = new_session("isnet-general-use")
    output = remove(input, bgcolor=(0, 0, 0, 255), session=session)
    output.save(output_path)


if __name__ == "__main__":
    main()
    # D:\02-code\01-github\U-2-Net\test_data\u2net_crops\ref 2_crop.png
    # img_array = np.array(Image.open(r'D:\02-code\01-github\U-2-Net\test_data\u2net_crops\ref 2_crop.png'))
    # print(img_array)
    pass
