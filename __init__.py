from nonebot import on_command
from nonebot.adapters.onebot.v11 import Bot, Event, Message, MessageSegment
from nonebot.typing import T_State
from nonebot.permission import SUPERUSER
from nonebot.adapters.cqhttp import PRIVATE
from nonebot.log import logger
import re
import os
import requests
from PIL import Image
from math import *

test1 = on_command('test', permission=SUPERUSER|PRIVATE)


@test1.handle()
async def test1_handle(bot: Bot, event: Event):
    await test1.send("blabla")

    #await test1.send(str(event.get_message()))
    img_cq = str(event.get_message())
    try:
        f_z = eval(re.search("{(.*)}", img_cq).group(1))
        img_url = re.search("url=(.*)]", img_cq).group(1)
    except AttributeError:
        await test1.send("C")
        return
    if not os.path.exists("tmp"):
        os.mkdir("tmp")
    if not os.path.exists("tmp_out"):
        os.mkdir("tmp_out")
    img_path = "tmp/" + img_url.split("/")[-1]
    # out_path = "tmp_out/" + img_url.split("/")[-1]+'.png'
    out_path = "/home/ubuntu/go-cqhttp/data/images/" + img_url.split("/")[-1] + '.png'
    out_name = img_url.split("/")[-1] + '.png'
    picfile = requests.get(img_url)
    open(img_path, 'wb').write(picfile.content)

    import numpy as np
    import cv2, time
    import multiprocess as multiprocessing
    import matplotlib.pyplot as plt

    def inverse(img):
        b, g, r = cv2.split(img)
        b = 255 - b
        g = 255 - g
        r = 255 - r
        img[:, :, 0] = b
        img[:, :, 1] = g
        img[:, :, 2] = r
        return img

    def transform(x, y, orgX, orgY, f=lambda z: z ** 1.2):
        c = complex(x - orgX, y - orgY)
        try:
            f_c = f(c)
            return f_c
        except ZeroDivisionError:
            return 0
        #return f(c)

    global const
    const = np.array([256, 256, 256], np.int16)

    def toMatrix(newDict):
        global const
        arrs = newDict.keys()
        xRange = max(arrs, key=lambda x: x[0])[0] - min(arrs, key=lambda x: x[0])[0]
        yRange = max(arrs, key=lambda x: x[1])[1] - min(arrs, key=lambda x: x[1])[1]
        print("Rendering image of size {}x{}...".format(xRange, yRange))
        shiftX = xRange // 2
        shiftY = yRange // 2
        imgArr = np.zeros((yRange, xRange, 3), np.int16)
        for x in range(xRange):
            for y in range(yRange):
                imgArr[y, x, :] = np.array(newDict.get((x - shiftX, y - shiftY), [255, 255, 255]), np.int16)
        return const - imgArr

    def bgrTorgb(img):
        img_rgb = np.zeros(img.shape, img.dtype)
        img_rgb[:, :, 0] = img[:, :, 2]
        img_rgb[:, :, 1] = img[:, :, 1]
        img_rgb[:, :, 2] = img[:, :, 0]
        return img_rgb

    def show(ori, img, path):
        plt.subplot(121)
        plt.title('Original Image')
        plt.imshow(bgrTorgb(ori))
        plt.subplot(122)
        plt.title('Destination Image')
        plt.imshow(bgrTorgb(img))
        #plt.show()
        plt.savefig(path)

    def avPixels(newImg, m, n, bgr, c):
        a = round(m)
        b = round(n)
        for i in range(a - c, a + c):
            for j in range(b - c, b + c):
                if newImg.get((i, j)) is None:
                    newImg[(i, j)] = bgr

    def calculateSparseArray(img, wStart, wEnd, h, orgX, orgY, kernel):
        c = kernel // 2
        newImg = {}
        for x in range(wStart, wEnd):
            for y in range(h):
                xy = transform(x, y, orgX, orgY, f_z)

                avPixels(newImg, xy.real, xy.imag, img[y, x, :], c)
        return newImg

    img = cv2.imread(img_path)
    print('Image loaded, size: {}x{}'.format(img.shape[1], img.shape[0]))
    short_side = min(img.shape[0], img.shape[1])
    if short_side > 512:
        scale_percent = 512 / short_side  # percent of original size
    else:
        scale_percent = 1
    width = int(img.shape[1] * scale_percent)
    height = int(img.shape[0] * scale_percent)
    dim = (width, height)
    print('Image input size: {}x{}'.format(dim[0], dim[1]))
    # resize image
    img = cv2.resize(img, dim)

    # img = bgrTorgb(img)
    # plt.waitforbuttonpress()
    #t = time.time()
    height, width = img.shape[0:2]
    orgX, orgY = (width // 2, height // 2)
    kernel = 7
    threads = 6
    wPart = width // threads
    results = []
    pool = multiprocessing.Pool(processes=threads)
    for i in range(threads - 1):
        results.append(
            pool.apply_async(calculateSparseArray, (img, wPart * i, wPart * (i + 1), height, orgX, orgY, kernel,)))
    results.append(
        pool.apply_async(calculateSparseArray, (img, wPart * (threads - 1), width, height, orgX, orgY, kernel,)))
    pool.close()
    pool.join()
    print('Calculated the sparse matrices')
    #t = time.time()
    d1 = results[0].get()
    for i in range(1, len(results)):
        d1.update(results[i].get())
    #print('It takes {}s to merge the sparse matrices'.format(round(time.time() - t, 2)))
    #t = time.time()
    imgArr = toMatrix(d1)
    #print('It takes {}s to convert sparse matrices to a complete numpy three dimensional array'.format(
    #    round(time.time() - t, 2)))
    out_path = "/home/ubuntu/go-cqhttp/data/images/" + 'tmp.png'
    #cv2.imwrite(out_path, inverse(imgArr))
    show(img, inverse(imgArr), out_path)
    #print("file:///home/ubuntu/go-cqhttp/data/images/" + img_url.split("/")[-1] + '.png')
    img01 = MessageSegment.image("file:///home/ubuntu/go-cqhttp/data/images/" + 'tmp.png')
    #cqcode = f"[CQ:image,file={out_name}, id=40000]"
    await test1.send(img01)
        #Message(
        #f"[CQ:image,file={img_url}, id=40000]"))
        #file:///{out_path},id=40000]"))
