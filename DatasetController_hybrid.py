import numpy as np
import pandas as pd
import time
import os
from PIL import Image
import collections
import csv
import copy
import random
import scipy.ndimage
import chainer.functions as F
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from image_transformer import imageTransformer
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from skimage import transform
from Gdataset import Gdataset

from local_config import config

locals().update(config)


def create_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


class DatasetController:
    def __init__(self, batch_size, sequence_size, string_size=10, shuffle_att=False):
        self.tasks = tasks
        self.multi_tasks = [6001, 6002, 6003, 6004, 6005, 6006, 6007, 6008]
        self.single_tasks = [6001, 6002, 6003, 6004, 6005, 6006, 6007, 6008, 6009]
        self.batch_size = batch_size
        self.sequence_size = sequence_size
        self.num_channels = num_channels
        self.image_size = 128
        self.csv_col_num = csv_col_num
        self.shuffle_att = shuffle_att
        self.cameras = cameras
        self.attention_size = 28
        self.train_percentage = 0.8

        self.transformer = imageTransformer("empty")
        self.mixer = Gdataset(self.image_size)
        self.source_dataset_path = dataset_path
        self.dest_dataset_path = self.source_dataset_path
        self.source_multi_object_dataset_path = multi_object_dataset_path
        self.source_single_object_dataset_path = single_object_dataset_path

        self.out_filepath_attention = './processed_inputs_attention_28_new_model/'

        self.multi_task_desc = {
            'push white plate from left to right' : 6001,
            'push red bowl from left to right' : 6007,
            'push blue box from left to right': 6002,
            'push black-white qr-box from left to right': 6003,
            'pick-up white towel': 6004,
            'pick-up white plate' : 6001,
            'pick-up red bubble-wrap': 6005,
            'pick-up blue ring': 6006,
            'pick-up red bowl': 6007,
            'pick-up black dumble': 6008,
        }

        self.annotations = {}
        self.reverse_annotations = {}
        self.multi_map = {}
        self.string_size = string_size
        self.annotated_tasks = [5001, 5002]
        self.bag_of_words = self.fill_bag_of_words(self.annotated_tasks)
        for task in self.annotated_tasks:
            self.read_annotations(task)

        self.num_words = len(self.bag_of_words.keys())
        self.step_size = 5

        self.objects_describtors = {"white" : 1,
                        "blue": 2,
                        "black-white": 3,
                        "black": 4,
                        "red": 5}
        self.objects = {"plate": 1,
                        "box": 2,
                        "qr-box": 3,
                        "bubble-wrap": 4,
                        "bowl": 5,
                        "towel": 6,
                        "dumble": 7,
                        "ring": 8}
                        
        self.objects_describtors = {self.bag_of_words[x]:y for x,y in self.objects_describtors.items()}
        self.objects = {self.bag_of_words[x]:y for x,y in self.objects.items()}
        self.num_all_objects = len(list(self.objects.keys()))
        self.num_all_objects_describtors = len(list(self.objects_describtors.keys()))

        self.train_folders = collections.defaultdict(dict)
        self.test_folders = collections.defaultdict(dict)
        self.train_multi_folders = {}
        self.test_multi_folders = {}
        self.train_single_folders = {}
        self.test_single_folders = {}
        self.separate_train_train()
        
        self.sentence_to_one_hot = {}
        self.sentence_to_objs = {}
        self.sentence_to_descs = {}
        for sentence in self.annotations:
            sentence_one_hot = np.zeros((self.max_sentence_len, self.num_words))
            for r, num in enumerate(sentence.split()):
                if int(num) > 0:
                    sentence_one_hot[r, int(num) - 1] = 1 
                
                if not sentence in self.sentence_to_objs and int(num) in self.objects:
                    self.sentence_to_objs[sentence] = self.objects[int(num)]
                if not sentence in self.sentence_to_descs and int(num) in self.objects_describtors:
                    self.sentence_to_descs[sentence] = self.objects_describtors[int(num)]
            
            self.sentence_to_one_hot[sentence] = sentence_one_hot

    def separate_train_train(self):
        for task_id in self.annotated_tasks:
            folder_annot = collections.defaultdict(list)
            for key in self.reverse_annotations[int(task_id)]:
                folder_annot[self.reverse_annotations[int(task_id)][key]].append(key)

            for key in folder_annot.keys():
                random.shuffle(folder_annot[key])
                folders = folder_annot[key]
                num_folders = len(folders)
                self.train_folders[str(task_id)][key] = folders[:int(num_folders * self.train_percentage)]
                self.test_folders[str(task_id)][key] = folders[int(num_folders * self.train_percentage):]
        
        for task_id in self.multi_tasks:
            task_path = os.path.join(self.source_multi_object_dataset_path, str(task_id))
            images_names = [name for name in os.listdir(os.path.join(task_path, "1", "camera-1"))]

            num_images = len(images_names)
            self.train_multi_folders[task_id] = images_names[:int(num_images * self.train_percentage)]
            self.test_multi_folders[task_id] = images_names[int(num_images * self.train_percentage):]

        for task_id in self.single_tasks:
            task_path = os.path.join(self.source_single_object_dataset_path, str(task_id))
            images_names = [name for name in os.listdir(os.path.join(task_path, "1", "camera-1"))]

            num_images = len(images_names)
            self.train_single_folders[task_id] = images_names[:int(num_images * self.train_percentage)]
            self.test_single_folders[task_id] = images_names[int(num_images * self.train_percentage):]

    def encode_annotation(self, sentence):
        words = sentence.split()
        encoded = ''
        for i, word in enumerate(words):
            encoded += str(self.bag_of_words[word]) + ' '
            
        return encoded + '0'

    def read_annotations(self, task):
        self.reverse_annotations[task] = {}
        self.max_sentence_len = 0

        with open(os.path.join(self.source_dataset_path, str(task) + '_task_annotation.csv'), 'r') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in spamreader:
                words = row[0].split()
                key = ''
                key_len = 0
                for word in words:
                    key += str(self.bag_of_words[word]) + ' '
                    key_len += 1
                key += '0'
                key_len += 1
                if key_len > self.max_sentence_len:
                    self.max_sentence_len = key_len
                self.annotations[key] = row[1:]
                for dem in row[1:]:
                    self.reverse_annotations[task][dem] = key
                    self.multi_map[dem] = self.multi_task_desc[row[0]]

    def fill_bag_of_words(self, tasks):
        unique_words = []
        max_len = 0
        bag = {}
        for task in tasks:
            with open(os.path.join(self.source_dataset_path, str(task) + '_task_annotation.csv'), 'r') as csvfile:
                spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
                for row in spamreader:
                    words = row[0].split()
                    if len(words) > max_len:
                        max_len = len(words)
                    for word in words:
                        if word not in unique_words:
                            unique_words.append(word)

        for i, word in enumerate(unique_words):
            bag[word] = i + 1

        if max_len + 1 > self.string_size:
            print("ERROR: provided string size is smaller than the biggest annotation!")

        return bag

    def get_random_dem_single(self, task, train=True):
        if train:
            folders =  self.train_single_folders[task]
        else:
            folders =  self.test_single_folders[task]

        num_dems = len(folders)
        rand_dems = np.random.randint(num_dems, size=self.sequence_size)

        return [folders[x] for x in rand_dems]

    def get_random_dem_multi(self, task, train=True):
        if train:
            folders =  self.train_multi_folders[task]
        else:
            folders =  self.test_multi_folders[task]

        num_dems = len(folders)
        rand_dems = np.random.randint(num_dems, size=self.sequence_size)

        return [folders[x] for x in rand_dems]
        
    def get_random_demonstration(self, task_id, batch_index, train=True):
        sentence_keys = {'5001': ['1 2 3 0', '1 9 11 0', '1 2 8 0', '1 9 10 0', '1 6 7 0', '1 4 5 0'],
                         '5002': ['12 2 3 13 14 15 16 0', '12 9 11 13 14 15 16 0', '12 18 19 13 14 15 16 0', '12 4 17 13 14 15 16 0']}
        if task_id is None:
            rand_task = np.random.randint(len(self.tasks), size=1)[0]
            task_id = self.tasks[rand_task]
        elif type(task_id) == list:
            rand_task = np.random.randint(len(task_id), size=1)[0]
            task_id = task_id[rand_task]
        if train:
            keys = sentence_keys[task_id]
            i = np.random.randint(1000, size=1)[0] % len(keys)
            folders =  self.train_folders[task_id][keys[i]]  
        else:
            keys = sentence_keys[task_id]
            i = np.random.randint(1000, size=1)[0] % len(keys)
            folders =  self.test_folders[task_id][keys[i]]

        num_dems = len(folders)
        rand_dem = np.random.randint(num_dems, size=1)[0]

        return task_id, folders[rand_dem]

    def get_task_one_hot_vector(self, joints):
        one_hot = np.zeros((self.batch_size, len(self.tasks)), dtype=np.float32)
        for i in range(self.batch_size):
            one_hot[i][int(joints[i][0][1]) - 5001] = 1

        return one_hot

    def get_attention_label(self, task, dem_index):
        correct_sentence = self.reverse_annotations[int(task)][dem_index]
        labels = correct_sentence.split()
        key_toRet = np.zeros((self.string_size), dtype=int)
        which_describtor = 0
        which_object = 0

        which_object = self.sentence_to_objs[correct_sentence]
        which_describtor = self.sentence_to_descs[correct_sentence]

        return key_toRet, which_object, which_describtor

    def get_next_batch(self, task=None, channel_first=True, from_start=False, from_start_prob=0.1, train=True, camera='camera-1', return_multi=True, return_single=True, return_dem=True, return_mix=False):

        while True:
            original_robot_images = np.empty(
                (self.batch_size, self.sequence_size, self.num_channels, self.image_size, self.image_size))
            original_robot_mix_images = np.empty(
                (self.batch_size, self.sequence_size, self.num_channels, self.image_size, self.image_size))
            original_robot_single_images = np.empty(
                (self.batch_size, self.sequence_size, self.num_channels, self.image_size, self.image_size))
            original_robot_single_mix_images = np.empty(
                (self.batch_size, self.sequence_size, self.num_channels, self.image_size, self.image_size))
            original_robot_multi_images = np.empty(
                (self.batch_size, self.sequence_size, self.num_channels, self.image_size, self.image_size))
            original_robot_multi_single_images = np.empty(
                (self.batch_size, self.sequence_size, self.num_channels, self.image_size, self.image_size))
            batch_joints = np.empty((self.batch_size, self.sequence_size, self.csv_col_num))

            batch_sentences = np.zeros((self.batch_size, self.max_sentence_len, self.num_words))

            object_involved = np.zeros((self.batch_size))
            describtors_involved = np.zeros((self.batch_size))

            object_involved_one_hot = np.zeros((self.batch_size, self.num_all_objects))
            describtors_involved_one_hot = np.zeros((self.batch_size, self.num_all_objects_describtors))
            for i in range(self.batch_size):
                task_id, dem_index = self.get_random_demonstration(task, i, train=train)
                sentence = self.reverse_annotations[int(task_id)][str(dem_index)]

                # for r, num in enumerate(sentence.split()):
                #     if int(num) > 0:
                #         batch_sentences[i, r, int(num) - 1] = 1
                batch_sentences[i] =  self.sentence_to_one_hot[sentence]

                joints = np.load(os.path.join(self.dest_dataset_path, task_id, str(dem_index) + '-joints.npy'))

                joints_len = len(joints)
                last_joint = np.expand_dims(joints[-1], axis=0)
                last_joint = np.repeat(last_joint, self.sequence_size * self.step_size, axis=0)
                joints = np.concatenate((joints, last_joint), axis=0)

                if joints_len > ((self.sequence_size) * self.step_size):
                    robot_rand_index = np.random.randint(joints_len - ((self.sequence_size) * self.step_size), size=1)[0]
                else:
                    robot_rand_index = 0

                coin_toss = random.uniform(0, 1)
                if coin_toss < from_start_prob:
                    from_start = True
                if from_start:
                    robot_rand_index = 0
                robot_images_start_index = robot_rand_index

                if return_single:
                    coin_toss = random.uniform(0, 1)
                    if coin_toss < 1.0:
                        single_pics = self.get_random_dem_single(6009, train=train)
                        for p, pic in enumerate(single_pics):
                            single_image, single_image_mix, _, _ = self.read_image_single_empty(os.path.join(self.source_single_object_dataset_path, '6009', '1', 'camera-1', pic), dem_index)
                            original_robot_single_images[i, p] = self.pre_process_image(single_image)
                            original_robot_single_mix_images[i, p] = self.pre_process_image(single_image_mix)
                    else:
                        single_pics = self.get_random_dem_single(self.multi_map[str(dem_index)], train=train)
                        for p, pic in enumerate(single_pics):
                            single_image, single_image_mix = self.read_image_single(os.path.join(self.source_single_object_dataset_path, str(self.multi_map[str(dem_index)]), '1', 'camera-1',pic), dem_index)
                            original_robot_single_images[i, p] = self.pre_process_image(single_image)
                            original_robot_single_mix_images[i, p] = self.pre_process_image(single_image_mix)

                if return_multi:
                    multi_pics = self.get_random_dem_multi(self.multi_map[str(dem_index)], train=train)
                    for p, pic in enumerate(multi_pics):
                        multi_image = self.read_image_multi(os.path.join(self.source_multi_object_dataset_path, str(self.multi_map[str(dem_index)]), '1', 'camera-1',pic), dem_index)
                        original_robot_multi_images[i, p] = self.pre_process_image(multi_image)

                if return_dem:
                    original_robot_images[i], original_robot_mix_images[i], camera_used = self.read_robot_npy_batch(task_id, dem_index, camera,
                                                            joints[robot_images_start_index: robot_images_start_index + self.step_size * self.sequence_size, 0], return_mix=return_mix)
                batch_joints[i] = joints[robot_rand_index: robot_rand_index + (self.sequence_size) * self.step_size: self.step_size]

                if int(task_id) in self.annotated_tasks:
                    label, which_object, which_describtor = self.get_attention_label(task_id, dem_index)

                    object_involved[i] = which_object
                    object_involved_one_hot[i, which_object - 1] = 1
                    describtors_involved[i] = which_describtor
                    describtors_involved_one_hot[i, which_describtor - 1] = 1

            
            batch_one_hot = self.get_task_one_hot_vector(batch_joints)
            to_ret_original_robot_images = original_robot_images
            to_ret_original_robot_mix_images = original_robot_mix_images
            to_ret_original_robot_single_images = original_robot_single_images
            to_ret_original_robot_single_mix_images = original_robot_single_mix_images
            to_ret_original_robot_multi_images = original_robot_multi_images

            yield to_ret_original_robot_images, to_ret_original_robot_mix_images,\
                to_ret_original_robot_single_images, to_ret_original_robot_single_mix_images,\
                to_ret_original_robot_multi_images,\
                batch_joints[:, :, 3:], batch_one_hot, batch_sentences, \
                object_involved, object_involved_one_hot, describtors_involved, describtors_involved_one_hot

    def read_image_multi(self, path, dem_index):
        image = Image.open(path)
        camera_id = 1
        noise = np.random.randint(20, size=2) - 10
        image = self.transformer.apply_homography(image, camera_id, noise=noise)
        return image
    
    def read_image_single_empty(self, path, dem_index):
        empty = Image.open(path)
        im2, _, image, multi_all_same, _ = self.mixer.mix_from_empty(empty, self.multi_map[dem_index] - 6000)
        camera_id = 1
        noise = np.random.randint(20, size=2) - 10
        im2 = self.transformer.apply_homography(im2, camera_id, noise=noise)
        image = self.transformer.apply_homography(image, camera_id, noise=noise)
        empty = self.transformer.apply_homography(empty, camera_id, noise=noise)
        multi_all_same = self.transformer.apply_homography(multi_all_same, camera_id, noise=noise)
        return im2, image, multi_all_same, empty

    def read_image_single(self, path, dem_index):
        image = Image.open(path)
        im2 = self.mixer.mix_image(copy.copy(image), self.multi_map[dem_index] - 6000)
        camera_id = 1
        noise = np.random.randint(20, size=2) - 10
        im2 = self.transformer.apply_homography(im2, camera_id, noise=noise)
        image = self.transformer.apply_homography(image, camera_id, noise=noise)
        return image, im2

    def read_robot_npy_batch(self, task_id, dem_index, camera, timestamps, return_mix=False):
        if camera == 'random':
            cam_num = np.random.randint(len(self.cameras), size=1)[0]
            camera = 'camera-' + str(cam_num)

        images = np.empty((self.sequence_size, self.num_channels, self.image_size, self.image_size))
        images_mix = np.empty((self.sequence_size, self.num_channels, self.image_size, self.image_size))
        npys_path = os.path.join(self.dest_dataset_path, task_id, dem_index, 'robot', camera)
        noise = np.random.randint(20, size=2) - 10
        prev = None
        for i in range(0, len(timestamps), self.step_size):
            # print ts
            path = os.path.join(npys_path, str(int(timestamps[i])) + '.jpg')
            im = self.read_image(path)
            if return_mix:
                im2, prev = self.mixer.mix_image(copy.copy(im), self.multi_map[dem_index] - 6000, apply_previous=prev)
                im2 = self.transformer.apply_homography(im2, int(camera[-1]), noise=noise)
                im2 = self.pre_process_image(im2)
                image2 = np.asarray(im2, dtype=np.float32)
                images_mix[int(i / self.step_size)] = image2
            # im = Image.fromarray(im.astype('uint8'), "RGB")
            im = self.transformer.apply_homography(im, int(camera[-1]), noise=noise)
            im = self.pre_process_image(im)
            image = np.asarray(im, dtype=np.float32)
            images[int(i / self.step_size)] = image

        return images, images_mix, camera

    def read_image(self, path):
        image = Image.open(path)
        return image

    def pre_process_image(self, image):
        if self.num_channels == 1:
            image = image.convert('L')

        if self.num_channels == 3:
            image = image.convert('RGB')
            # image = np.array(image)
            # image = image.transpose((2, 0, 1))

        image = np.asarray(image, dtype=np.float32)
        image = image[:, ::-1, :].copy()
        image = image.transpose((2, 0, 1))
        image = image / 127.5 - 1
        # image = image / 255
        return image

def show_image(images):
    # for i in range(images.shape[0]):
        # moving axis to use plt: i.e [4,100,100] to [100,100,4]
    img = images
    img = img.transpose(1, 2, 0)
    img = (img + 1) *127.5
    img = img.astype(np.uint8)
    print(img.dtype, np.max(img), np.min(img), np.shape(img))
    img = Image.fromarray(img, "RGB")
    img.show()

def save_image(data, filename, num_channels, image_size):
        image = ((data + 1) * 128).clip(0, 255).astype(np.uint8)
        sample_image_rows = 4
        sample_image_cols = 8
        image = image[:sample_image_rows * sample_image_cols]
        image = image.reshape(
            (sample_image_rows, sample_image_cols, num_channels, image_size,
                image_size)).transpose(
            (0, 3, 1, 4, 2)).reshape(
            (sample_image_rows * image_size, sample_image_cols * image_size, num_channels))
        Image.fromarray(image).save(filename)

if __name__ == '__main__':
    DC = DatasetController(batch_size=50, sequence_size = 5, read_jpgs=True)
    # g = DC.get_next_batch(task='5002', channel_first=False, human=False, sth_sth=True, joint_history=10)
    g1 = DC.get_next_batch(task=['5001', '5002'], channel_first=True, from_start_prob=0, camera='camera-1', return_mix=True, return_single=True, return_dem=True, return_multi=True)

    # while True:
    start = time.time()
    # robot_images, robot_paths, human_images, human_paths, sth_sth_images, sth_sth_paths, joints, noisy_joints, batch_one_hot, attention_labels, attention_gt = next(g)
    to_ret_robot_images_orginals, to_ret_robot_mix_images_orginals, \
    to_ret_robot_single_images_orginals, to_ret_robot_single_mix_images_orginals, \
    to_ret_robot_multi_images_orginals, \
    batch_joints, batch_one_hot, batch_sentences, \
    objects, obj_one_hot, descriptions, desc_one_hot = next(g1)
    save_image(to_ret_robot_images_orginals[:, 0], './paper_figure/robot.png', 3, 128)
    save_image(to_ret_robot_mix_images_orginals[:, 0], './paper_figure/robot_mix.png', 3, 128)
    save_image(to_ret_robot_single_images_orginals[:, 0], './paper_figure/fake.png', 3, 128)
    save_image(to_ret_robot_single_mix_images_orginals[:, 0], './paper_figure/fake_mix.png', 3, 128)
    save_image(to_ret_robot_multi_images_orginals[:, 0], './paper_figure/real.png', 3, 128)

    print((time.time() - start))
