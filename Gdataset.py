import numpy as np
import os
import copy
import collections
from PIL import Image
import random

from local_config import config

locals().update(config)

class Gdataset(object):
    def __init__(self, image_size):
        self.dataset_path = dataset_path
        self.objects_path = './objects_grid/proccessed/'
        self.image_size = image_size
        self.object_map = { 1: 'white plate',
                            2: 'blue box',
                            3: 'black-white qr-box',
                            4: 'white towel',
                            5: 'red bubble-wrap',
                            6: 'blue ring',
                            7: 'red bowl',
                            8: 'black dumble'}
        self.read_object_images()
        self.obj_range = set([x + 1 for x in range(8)])

    def read_object_images(self):
        self.object_images = collections.defaultdict(list)
        for subdir_name in os.listdir(self.objects_path):
            for img_name in range(1, 17):
                self.object_images[int(subdir_name)].append(self.read_image(os.path.join(self.objects_path, subdir_name, str(img_name) + '.png')))

    def mix_from_empty(self, empty, obj_in_img, other=None):
        orig_cell, other_cell, multi_cell = np.random.choice(range(1, 17), size=3, replace=False)
        toMergeOrig = self.object_images[obj_in_img][orig_cell - 1]
        wAdd = ((orig_cell - 1) % 4) * self.image_size / 5
        wAdd += np.random.randint(self.image_size//16, size=1)[0] - self.image_size//32
        hAdd = ((orig_cell - 1) // 4) * self.image_size / 5
        hAdd += np.random.randint(self.image_size//16, size=1)[0] - self.image_size//32
        hAdd, wAdd = int(hAdd), int(wAdd)
        degree = np.random.randint(30, size=1) - 15
        toMergeOrig = toMergeOrig.rotate(int(degree), resample=Image.BICUBIC)

        single_1 = copy.copy(empty)
        multi_all_same = copy.copy(empty)
        single_1.paste(toMergeOrig, (wAdd, hAdd), mask=toMergeOrig)
        multi_all_same.paste(toMergeOrig, (wAdd, hAdd), mask=toMergeOrig)


        single_2 = copy.copy(empty)
        if other == None:
            s = random.sample(self.obj_range - set([obj_in_img]), 1)[0]
        else:
            s = other

        toMerge = self.object_images[s][other_cell - 1]
        wAdd = ((other_cell - 1) % 4) * self.image_size / 5
        wAdd += np.random.randint(self.image_size//16, size=1)[0] - self.image_size//32
        hAdd = ((other_cell - 1) // 4) * self.image_size / 5
        hAdd += np.random.randint(self.image_size//16, size=1)[0] - self.image_size//32
        hAdd, wAdd = int(hAdd), int(wAdd)
        degree = np.random.randint(30, size=1) - 15
        toMerge = toMerge.rotate(int(degree), resample=Image.BICUBIC)
        toMergeOrig = toMergeOrig.rotate(int(degree), resample=Image.BICUBIC)

        single_2.paste(toMerge, (wAdd, hAdd), mask=toMerge)
        multi_all_same.paste(toMergeOrig, (wAdd, hAdd), mask=toMergeOrig)
        multi = copy.copy(single_1)

        multi.paste(toMerge, (wAdd, hAdd), mask=toMerge)
        return single_1, single_2, multi, multi_all_same, s

    def mix_image(self, im, obj_in_img, can_zero=False, apply_previous=None):
        if apply_previous:
            for i, s in enumerate(apply_previous):
                im.paste(s[0], (s[1], s[2]), mask=s[0])
            return im, apply_previous

        if can_zero:
            num = random.randint(0,2)
        else:
            num = random.randint(1,2)

        selected = random.sample(self.obj_range - set([obj_in_img]), num)
        w, h = im.size

        cells = np.random.choice(range(1, 17), size=num, replace=False)
        applied_images = []
        for i, s in enumerate(selected):
            toMerge = self.object_images[s][cells[i] - 1]
            wM, hM = toMerge.size
            wAdd = ((cells[i] - 1) % 4) * self.image_size / 5
            wAdd += np.random.randint(self.image_size//16, size=1)[0] - self.image_size//32
            hAdd = ((cells[i] - 1) // 4) * self.image_size / 5
            hAdd += np.random.randint(self.image_size//16, size=1)[0] - self.image_size//32
            hAdd, wAdd = int(hAdd), int(wAdd)
            degree = np.random.randint(30, size=1) - 15
            toMerge = toMerge.rotate(int(degree), resample=Image.BICUBIC)
            applied_images.append((toMerge, wAdd, hAdd))
            im.paste(toMerge, (wAdd, hAdd), mask=toMerge)

        return im, applied_images

    def read_image(self, path):
        image = Image.open(path)
        ratio = 224.0 / self.image_size
        image = image.resize((int(image.size[0]/ratio), int(image.size[1]/ratio)), Image.ANTIALIAS)
        return image

# DG = Gdataset(128)
# DG.mix_image(Image.open('/home/d3gan/development/datasets/record/sth_sth_128/5001/4766/robot/camera-1/15323594758.jpg'), 1)
