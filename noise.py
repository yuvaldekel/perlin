from itertools import product
from math import sin, cos
import numpy as np
import os
from PIL import Image
from random import randint
import sys
from thread_with_return import ThreadWithReturn
import time

PATH = r"/home/yuval/Documents/yuval/Devops-linux/python/perlin/images"
Q = 0.5
C = 2
LAYERS = 10


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


    def apply_algorithms(self, part):
        
        if part == 1:
        
            x = self.array_x[:800]
            y = self.array_y[:800]

        else:
            x = self.array_x[800:]
            y = self.array_y[800:]


        apply_all = np.vectorize(self.perlin)
        return apply_all(x, y)


def normalize(array):
    array_min = abs(np.min(array))
    array_max = np.max(array)

    abs_max = max(array_min, array_max)

    array = array / abs_max
    array = (array + 1) * 127.5 
    return array


def save_img(img):
    images = list(filter(lambda file: file.endswith(".PNG"), os.listdir(PATH)))
    
    if not images:
        next = 0

    else:    
        current = max(images)
    
        start = 6
        end = current.find('.')
        next = int(current[start:end]) + 1


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


def main():
    pixels = int(sys.argv[1])
    frequency =  pixels / C
    corners = C
    image_array = None

    #while corners < pixels:
    for i in range(LAYERS):
        print(i)
        perlin_instance = PerlinNoise(corners, corners, frequency)
        #current_image_array = perlin_instance.create_array(frequency)

        t1 = ThreadWithReturn(target=perlin_instance.apply_algorithms, args=(1,))
        t2 = ThreadWithReturn(target=perlin_instance.apply_algorithms, args=(2,))

        t1.start()
        t2.start()

        part1 = t1.join()
        part2 = t2.join()

        #print(part1.shape)
        #print(part2.shape)

        current_image_array = np.append(part1, part2, axis = 0)
        
        if image_array is None:
            image_array = current_image_array

        else:
            w = (1 * (Q ** i))
            sum_w = (Q ** (i) - 1) / -Q

            image_array = image_array * sum_w + w * current_image_array 
            image_array = image_array / (sum_w + w) 

        new_frequency = frequency / 2
        corners = corners * 2

        frequency = new_frequency
    
    image_array = normalize(image_array)

    image_array = np.asarray(image_array)
    image_array = np.asarray(image_array)
    img = Image.fromarray(image_array)

    save_img(img)

if __name__ == "__main__":
    time_program(main)