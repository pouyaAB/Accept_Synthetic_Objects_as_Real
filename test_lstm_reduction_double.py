import numpy as np
import time
import sys
import csv
from local_config import config
import scipy.ndimage

locals().update(config)

sys.path.append("/home/d3gan/catkin_ws/src/ros_teleoprtate/al5d/scripts/")

from test_network_robot import TestNetworkRobot
import os
import copy
import chainer.functions as F
from PIL import Image
import threading
import signal
import copy

from sklearn.decomposition import PCA
from sklearn.externals import joblib

import tensorflow as tf
from gpu import GPU
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
import autoencoders.reduction_att
import autoencoders.tower
# from nf_mdn_rnn import MDN_RNN
from nf_mdn_rnn import RobotController
import matplotlib.pyplot as plt
import cv2

from DatasetController_hybrid import DatasetController


def signal_handler(signal, frame):
    global MT
    MT.dataset_ctrl.end_thread = True
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


class ModelTester:
    """
        Handles the model interactions
    """

    def __init__(self, image_size, latent_size, output_size=7, num_channels=3, task='5002', tasks=['5001', '5002', '5003', '5004'], save_dir="model/reduction_double_works/"):
        self.image_size = image_size
        self.latent_size = latent_size
        self.num_channels = num_channels
        self.save_dir = save_dir
        self.output_size = output_size
        self.batch_size = 1
        self.task = task
        self.att_size = 16
        self.tasks = tasks
        self.attention_latent_size = 32
        self.attention_size = 8
        self.robot_latent_size = 4
        self.text_encoding_size = 32
        self.auto_regressive = False
        self.use_all_cameras = False
        self.use_all_cameras_stacked_on_channel = False
        if self.use_all_cameras_stacked_on_channel:
            self.num_channels = self.num_channels * 3
        self.which_camera_to_use = 1

        self.dataset_ctrl = TestNetworkRobot(self.image_size, config['record_path'], config['robot_command_file'],
                                             config['camera_topics'], cache_size=1,
                                             cameras_switch=[False, True, False])

        
        self.DC = DatasetController(batch_size=32, sequence_size=1, read_jpgs=True, shuffle_att=False)
        self.g = self.DC.get_next_batch(task=['5002', '5001'], from_start_prob=0, camera='camera-1', return_multi=True)

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

        self.num_descs = self.DC.num_all_objects_describtors
        self.num_objects = self.DC.num_all_objects
        self.enc_model = autoencoders.reduction_att.Encoder_text_tower(density=8, size=self.image_size, latent_size=latent_size, att_size=self.att_size, num_objects=self.num_objects, num_describtions=self.num_descs)
        self.att_enc_model = autoencoders.tower.Encoder(density=8, size=self.image_size, latent_size=latent_size)
        self.att_gen_model = autoencoders.reduction_att.Generator(density=8, size=image_size, latent_size=latent_size, att_size=self.att_size, channel=2 * self.num_channels, num_objects=self.num_objects, num_describtions=self.num_descs)
        self.mdn_model = RobotController(self.latent_size + self.num_objects + self.num_descs + 4, hidden_dimension, output_size, num_mixture_2, auto_regressive=False)

        self.annotated_tasks = [5001, 5002]

        # raw_sentence = "push blue box from left to right"
        raw_sentence = "pick-up white plate"
        self.obj_one_hot = np.zeros(self.num_objects, dtype=np.float32)
        self.desc_one_hot = np.zeros(self.num_descs, dtype=np.float32)

        for word in raw_sentence.split():
            if word in self.objects_describtors:
                self.desc_one_hot[self.objects_describtors[word] - 1] = 1
            if word in self.objects:
                self.obj_one_hot[self.objects[word] - 1] = 1

        self.sentence = self.DC.encode_annotation(raw_sentence)
        self.sentence = np.asarray(self.DC.sentence_to_one_hot[self.sentence], dtype=np.float32)

        self.load_model()
        self.to_gpu()
        self.real_time_test()

    def get_task_one_hot_vector(self, task):
        one_hot = np.zeros((self.batch_size, len(self.tasks)), dtype=np.float32)
        for i in range(self.batch_size):
            one_hot[i][int(task) - 5001] = 1

        return one_hot

    def real_time_test(self):
        predicted = np.zeros((self.output_size), dtype=np.float32)
        self.mdn_model.reset_state()
        seed = None
        count = 0
        while True:
            input_online_images, _ = self.dataset_ctrl.get_next_batch(batch_size=1, camera_id=self.which_camera_to_use, channel_first=True, homography=True)

            obj_one_hot = Variable(cuda.to_gpu(self.obj_one_hot[np.newaxis], GPU.main_gpu))
            desc_one_hot = Variable(cuda.to_gpu(self.desc_one_hot[np.newaxis], GPU.main_gpu))

            input_online_images = np.asarray(input_online_images[0], dtype=np.float32)
            x_in = cuda.to_gpu(input_online_images[np.newaxis], GPU.main_gpu)
            att, _, _ = self.enc_model(Variable(x_in), obj_one_hot, desc_one_hot, train=False)

            att = F.reshape(att, (-1, 1, self.att_size, self.att_size))
            att = F.resize_images(att, (self.image_size, self.image_size))

            masked = Variable(x_in) * att
            z, mean, var, _ = self.att_enc_model(masked, train=False)


            input_online_images = np.transpose(input_online_images, (1, 2, 0))
            tshow = cv2.cvtColor((input_online_images + 1) * 127.5, cv2.COLOR_BGR2RGB)
            cv2.imshow('Input image original(O)', cv2.resize(tshow.astype(np.uint8), (0,0), fx=2, fy=2, interpolation=cv2.INTER_NEAREST))
            cv2.moveWindow('Input image original(O)', 0, 0)
            cv2.waitKey(10)

            tshow = (cuda.to_cpu(masked.data)[0] + 1) * 127.5
            tshow = cv2.cvtColor(tshow.transpose((1, 2, 0)), cv2.COLOR_BGR2RGB)
            cv2.imshow('masked image(M)', cv2.resize(tshow.astype(np.uint8), (0,0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC))
            cv2.moveWindow('masked image(M)', 450, 0)
            cv2.waitKey(1)

            x_in_att = self.att_gen_model(z, train=False)

            tshow = (cuda.to_cpu(x_in_att[:, :3].data)[0] + 1) * 127.5
            tshow = cv2.cvtColor(tshow.transpose((1, 2, 0)), cv2.COLOR_BGR2RGB)
            cv2.imshow('Reconstruction whole image(O\')', cv2.resize(tshow.astype(np.uint8), (0,0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC))
            cv2.moveWindow('Reconstruction whole image(O\')', 850, 0)
            cv2.waitKey(1)

            tshow = (cuda.to_cpu(x_in_att[:, 3:].data)[0] + 1) * 127.5
            tshow = cv2.cvtColor(tshow.transpose((1, 2, 0)), cv2.COLOR_BGR2RGB)
            cv2.imshow('Reconstruction masked image(M\')', cv2.resize(tshow.astype(np.uint8), (0,0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC))
            cv2.moveWindow('Reconstruction masked image(M\')', 1250, 0)
            cv2.waitKey(1)

            task_one_hot = self.get_task_one_hot_vector(self.task)
            task_one_hot = Variable(cuda.to_gpu(task_one_hot, GPU.main_gpu))
            text_encoding = F.concat((task_one_hot, obj_one_hot, desc_one_hot), axis=-1)

            input_feature = z
            input_feature = F.expand_dims(input_feature, axis=0)

            if count % 60 == 0:
                self.mdn_model.reset_state()
                print '########################RESET####################################'

            dummy_joints = Variable(cuda.to_gpu(np.zeros((1, 1, self.output_size), dtype=np.float32), GPU.main_gpu))
            _, sample = self.mdn_model(task_encoding=text_encoding, image_encoding=input_feature, data_out=dummy_joints, return_sample=True, train=False)
            seed = sample
            predicted = cuda.to_cpu(sample.data)[0]
            count += 1
            self.dataset_ctrl.send_command(predicted)
            print(predicted)

    def show_image(self, images):
        for i in range(images.shape[0]):
            img = images[i]
            img = img.transpose(1, 2, 0)
            img = (img +1) *127.5
            img = img.astype(np.uint8)
            print(img.dtype, np.max(img), np.min(img), np.shape(img))
            img = Image.fromarray(img, "RGB")
            img.show()

    def write_command(self, command):
        with open(os.path.join(record_path, 'commands.csv'), 'wb') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow(command)

    def load_model(self):
        try:
            file_path = os.path.join(os.path.dirname(__file__), self.save_dir)

            serializers.load_hdf5(file_path + 'enc.model', self.enc_model)
            serializers.load_hdf5(file_path + 'att_enc.model', self.att_enc_model)
            serializers.load_hdf5(file_path + 'att_gen.model', self.att_gen_model)
            serializers.load_hdf5(file_path + 'rnn_mdn.model', self.mdn_model)
            print('Models has been loaded!')
        except Exception as inst:
            print(inst)
            print('cannot load model from {}'.format(file_path))
            sys.exit(0)

    def to_gpu(self):
        self.enc_model.to_gpu(GPU.main_gpu)
        self.att_enc_model.to_gpu(GPU.main_gpu)
        self.att_gen_model.to_gpu(GPU.main_gpu)
        self.mdn_model.to_gpu(GPU.main_gpu)

if __name__ == '__main__':
    global MT
    MT = ModelTester(image_size, latent_size, task=task, tasks=tasks, output_size=7)