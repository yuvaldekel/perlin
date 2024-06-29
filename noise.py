from glob import glob
from itertools import product
from math import sin, cos, sqrt
from multiprocessing import Process
from multiprocessing.managers import NamespaceProxy
import numpy as np
import os
from PIL import Image
from random import randint
import sys
import time

PATH = r"/home/yuval/Documents/yuval/Devops-linux/python/perlin/images"


class ProxyBase(NamespaceProxy):
    _exposed_ = ('__getattribute__', '__setattr__', '__delattr__')


class PerlinNoise:


    def init_random_gradient(self) -> None:
        self.corners_gradient = {}

        for corner_x in range(self.x_size + 1):

            for corner_y in range(self.y_size + 1):
                angle = randint(0, 359)
                vector_x = cos(angle)
                vector_y = sin(angle)

                self.corners_gradient[(corner_x, corner_y)] = (vector_x, vector_y)


    def create_array(self, frequency):
        x_pixels = int(self.x_size * frequency)
        y_pixels = int(self.y_size * frequency)

        array_x = np.arange(0, x_pixels) / frequency
        array_y = np.arange(0, y_pixels) / frequency
        pairs = np.array(list(product(array_x, array_y))).T

        self.array_x = pairs[0].reshape(x_pixels, y_pixels)
        self.array_y = pairs[1].reshape(x_pixels, y_pixels)


    def __init__(self, x_size, y_size, frequency) -> None:
        self.x_size = x_size 
        self.y_size = y_size
        self.result = {}
        self.init_random_gradient()
        self.create_array(frequency)


    @staticmethod
    def dot(vector1, vector2):
        return vector1[0] * vector2[0] + vector1[1] * vector2[1]


    @staticmethod
    def smothstep(w):
        return 3 * (w ** 2) -  2 *(w ** 3)


    def perlin(self, x, y):
        vectors = [0] * 4
        corner_vector = [0] * 4

        int_x = int(x)
        int_y = int(y)

        corner_vector[0] = self.corners_gradient[(int_x, int_y)]
        corner_vector[1] = self.corners_gradient[(int_x, int_y + 1)]
        corner_vector[2] = self.corners_gradient[(int_x + 1, int_y)]
        corner_vector[3] = self.corners_gradient[(int_x + 1, int_y + 1)]

        vectors[0] = (x - int_x, y - int_y)
        vectors[1] = (x - int_x, y - int_y - 1)
        vectors[2] = (x - int_x - 1, y - int_y)
        vectors[3] = (x - int_x - 1, y - int_y - 1)

        dot_lu = self.dot(corner_vector[0], vectors[0])
        dot_lb = self.dot(corner_vector[1], vectors[1])
        dot_ru = self.dot(corner_vector[2], vectors[2])
        dot_rb = self.dot(corner_vector[3], vectors[3])

        w = (y - int_y) 
        dot_l = dot_lu + self.smothstep(w) * (dot_lb - dot_lu)
        dot_r = dot_ru + self.smothstep(w) * (dot_rb - dot_ru)

        w = (x - int_x) 
        return dot_l + self.smothstep(w) * (dot_r - dot_l)


    def calc_pixel(self, x, y):
        result = self.perlin(x, y)
        gradient_y = (self.perlin(x, y + 0.001) - result) / 0.001
        gradient_x = (self.perlin(x + 0.001, y) - result) / 0.001
        gradient = sqrt(gradient_y ** 2 + gradient_x ** 2)

        return result, gradient


    def apply_algorithms(self):
        apply_all = np.vectorize(self.calc_pixel)
        return apply_all(self.array_x, self.array_y)

        
def normalize(array):
    array_min = abs(np.min(array))

    array += array_min
    
    array_max = np.max(array)

    return array / array_max * 255 


def save_img(img):
    images = list(filter(lambda file: file.endswith(".PNG"), os.listdir(PATH)))
    
    if not images:
        next = 0

    else:    
        next = max([int(image[6:image.find('.')]) for image in images]) + 1

    name = f"perlin{next}.PNG"
    img = img.convert("L")
    img.save(f"{PATH}/{name}")

    img = Image.open(f"{PATH}/{name}")
    img.show()


def time_program(function):
    start = time.perf_counter()
    function()
    end = time.perf_counter()

    print(f'{float(end - start):.8f}')


def process_init(i, corners, frequency):
    perlin_instance = PerlinNoise(corners, corners, frequency)
    pixels, gradients = perlin_instance.apply_algorithms()
    np.save(f"./array/pixels{i}.npy", pixels)
    np.save(f"./array/gradients{i}.npy", gradients)


def main():
    files = glob('./array/*')
    for f in files:
        os.remove(f)

    pixels = int(sys.argv[1])
    corners = int(sys.argv[2])
    layers = int(sys.argv[3])
    frequency =  pixels / corners

    image_array = None
    gradient_array = None
    processes = [] 

    #while corners < pixels:
    for i in range(layers):
        process_i = Process(target = process_init, args = [i, corners, frequency])
        process_i.start()
        processes.append(process_i)

        new_frequency = frequency / 2
        corners = corners * 2

        frequency = new_frequency

    for process in processes:
        process.join()
        process.close()

    for i in range(layers):
        current_image_array = np.load(f"./array/pixels{i}.npy")
        current_gradient_array = np.load(f"./array/gradients{i}.npy")

        if image_array is None:
            gradient_array = current_gradient_array
            image_array = current_image_array * (1 / (1 + gradient_array))

        else:
            amplitude = 0.5 ** i

            gradient_array += current_gradient_array * amplitude
            image_array += amplitude * current_image_array  * (1 / (1 + gradient_array))

    image_array = normalize(image_array)

    image_array = np.asarray(image_array)
    image_array = np.asarray(image_array)
    img = Image.fromarray(image_array)

    save_img(img)


if __name__ == "__main__":
    time_program(main)