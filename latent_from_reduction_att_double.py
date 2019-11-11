import numpy as np
import time
import sys
import os
import copy
import chainer.functions as F
import signal
import pandas as pd
from PIL import Image
import threading

from gpu import GPU
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
import autoencoders.reduction_att
import autoencoders.tower
from local_config import config
from image_transformer import imageTransformer

from DatasetController_hybrid import DatasetController

def signal_handler(signal, frame):
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

locals().update(config)


def create_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


class EncodeDataset:
    def __init__(self):
        self.model_path = "model/"
        self.out_filepath = './proccessed_data/'

        create_dir(self.out_filepath)

        self.image_size = image_size
        self.latent_size = latent_size
        self.batch_size = 200
        self.sequence_size = 1
        self.att_size = 16
        self.robot_latent_size = 4
        self.dataset_path = dataset_path
        self.source_dataset_path = dataset_path
        self.num_channels = num_channels
        self.cameras_to_process = ['camera-1']
        self.cameras = self.cameras_to_process
        self.tasks_to_process = ['5001', '5002']

        self.transformer = imageTransformer("empty")
        self.use_all_cameras_stacked_on_channel = False
        if self.use_all_cameras_stacked_on_channel:
            self.num_channels = self.num_channels * 3

        self.batch_gpu_threads = [None] * GPU.num_gpus

        self.DC = DatasetController(batch_size=int(batch_size/2), sequence_size=sequence_size, shuffle_att=False)

        self.num_descs = self.DC.num_all_objects_describtors
        self.num_objects = self.DC.num_all_objects
        self.text_encoding_size = self.num_objects + self.num_descs

        self.enc_models = [None] * 2
        self.att_enc_models = [None] * 2
        self.att_gen_models = [None] * 2

        self.att_enc_models[0] = autoencoders.tower.Encoder(density=8, size=self.image_size, latent_size=latent_size)
        self.att_enc_models[1] = autoencoders.tower.Encoder(density=8, size=self.image_size, latent_size=latent_size)

        self.enc_models[0] = autoencoders.reduction_att.Encoder_text_tower(density=8, size=self.image_size, latent_size=latent_size, att_size=self.att_size, num_objects=self.num_objects, num_descriptions=self.num_descs)
        self.enc_models[1] = autoencoders.reduction_att.Encoder_text_tower(density=8, size=self.image_size, latent_size=latent_size, att_size=self.att_size, num_objects=self.num_objects, num_descriptions=self.num_descs)

        self.att_gen_models[0] = autoencoders.reduction_att.Generator(density=8, size=image_size, latent_size=latent_size, att_size=self.att_size, channel=2 * self.num_channels, num_objects=self.num_objects, num_descriptions=self.num_descs)
        self.att_gen_models[1] = autoencoders.reduction_att.Generator(density=8, size=image_size, latent_size=latent_size, att_size=self.att_size, channel=2 * self.num_channels, num_objects=self.num_objects, num_descriptions=self.num_descs)

        self.load_model()
        self.to_gpu()

        self.process_dataset()

    def encode(self, image, obj, desc, num):
        xp = cuda.cupy
        cuda.get_device(GPU.gpus_to_use[num % GPU.num_gpus]).use()

        obj = np.asarray(obj, dtype=np.float32)
        obj = np.repeat(obj[np.newaxis], image.shape[0], axis=0)
        desc = np.asarray(desc, dtype=np.float32)
        desc = np.repeat(desc[np.newaxis], image.shape[0], axis=0)

        o_in = cuda.to_gpu(obj, GPU.gpus_to_use[num % GPU.num_gpus])
        d_in = cuda.to_gpu(desc, GPU.gpus_to_use[num % GPU.num_gpus])
        x_in = cuda.to_gpu(image, GPU.gpus_to_use[num % GPU.num_gpus])

        att, _, _ = self.enc_models[num%2](Variable(x_in), Variable(o_in), Variable(d_in), train=False)

        att = F.reshape(att, (-1, 1, self.att_size, self.att_size))
        att = F.resize_images(att, (self.image_size, self.image_size))

        cir_z, _, _, _ = self.att_enc_models[num%2](Variable(x_in) * att, train=False)

        return cir_z, F.squeeze(F.concat((o_in[0], d_in[0]), axis=-1))

    def process_dataset(self):
        for dir_name in os.listdir(self.source_dataset_path):
            if os.path.isdir(os.path.join(self.source_dataset_path, dir_name)):
                print(('Found directory: %s' % dir_name))
                if dir_name in self.tasks_to_process:
                    if int(dir_name) in list(self.DC.reverse_annotations.keys()):
                        self.process_tasks(dir_name)

    def process_tasks(self, dir_name):
        source = os.path.join(self.source_dataset_path, dir_name)
        dest = os.path.join(self.out_filepath, dir_name)
        create_dir(dest)

        for subdir_name in os.listdir(source):
            if subdir_name in list(self.DC.reverse_annotations[int(dir_name)].keys()):
                _, which_obj, which_desc = self.DC.get_attention_label(dir_name, subdir_name)

                object_involved_one_hot = np.zeros((1, self.num_objects), dtype=np.float32)
                describtors_involved_one_hot = np.zeros((1, self.num_descs), dtype=np.float32)
                object_involved_one_hot[0, which_obj - 1] = 1
                describtors_involved_one_hot[0, which_desc - 1] = 1

                threads = []
                # Generating 40 different combinations of synthetic objects for each demonstration
                # Starting from 10 so that all folders have 2 additional digits in their name
                for num in range(10, 51):
                    if num!= 0 and num % 2 == 0:
                        for t1 in threads:
                            t1.join()
                        threads = []

                    dest_subdir_path = os.path.join(dest, subdir_name + ':' + str(num))
                    create_dir(dest_subdir_path)
                    t = threading.Thread(target=self.process_demonstration, args=(os.path.join(source, subdir_name), subdir_name, dest_subdir_path, object_involved_one_hot, describtors_involved_one_hot, num))
                    threads.append(t)
                    t.start()
                
                for t in threads:
                    t.join()

    def process_demonstration(self, dir_path, dem_index, dest_dir_path, obj, desc, num):
        dem_folder = dir_path
        dem_csv = dir_path + '.txt'
        # print dem_csv
        joint_pos = pd.read_csv(dem_csv, header=2, sep=',', index_col=False)

        timestamps_robot = list(joint_pos['timestamp'])
        #Finding timesstamps that are present in all cameras
        for camera in self.cameras:
            _, _, valid_ts = self.images_to_latent(dem_folder, dem_index, camera, obj, desc, num, timestamps=timestamps_robot, verify=True)
            timestamps_robot = valid_ts

        for camera in self.cameras_to_process:
            if not os.path.exists(os.path.join(dest_dir_path, camera + '.npy')):
                latents, encodings, _ = self.images_to_latent(dem_folder, dem_index, camera, obj, desc, num ,timestamps=timestamps_robot, verify=False)
                np.save(os.path.join(dest_dir_path, camera + '.npy'), latents)
                np.save(os.path.join(dest_dir_path, 'encoding.npy'), encodings)

    def images_to_latent(self, path, dem_index, camera, obj, desc, num, timestamps=None, isRobot='robot', verify=False):
        valid_ts = []
        latents = np.empty((len(timestamps), self.latent_size))
        encodings = np.empty((len(timestamps), self.text_encoding_size))
        images = np.zeros((len(timestamps), self.num_channels, self.image_size, self.image_size), dtype=np.float32)
        noise = np.random.randint(20, size=2) - 10
        prev = None
        for i, ts in enumerate(timestamps):
            to_read = os.path.join(path, isRobot, camera, str(ts) + '.jpg')
            # print to_read
            if os.path.isfile(to_read):
                if not verify:
                    image = self.read_image(to_read, noise=noise)
                    image, prev = self.DC.mixer.mix_image(image, self.DC.multi_map[dem_index] - 6000, can_zero=False, apply_previous=prev)
                    image = self.pre_process_image(image)
                    images[i] = np.asarray(image, dtype=np.float32)
                valid_ts.append(ts)

        if not verify:
            if images.shape[0] > self.batch_size:
                for j in range(0, images.shape[0] - (images.shape[0] % self.batch_size), self.batch_size):
                    ext, encoding = self.encode(images[j:j + self.batch_size], obj, desc, num)
                    latents[j:j + self.batch_size] = cuda.to_cpu(ext.data)
                    encodings[j:j + self.batch_size] = cuda.to_cpu(encoding.data)

                j += self.batch_size
                if j < images.shape[0]:
                    ext, encoding = self.encode(images[j:], obj, desc, num)
                    latents[j:] = cuda.to_cpu(ext.data)
                    encodings[j:] = cuda.to_cpu(encoding.data)
            else:
                ext, encoding = self.encode(images, obj, desc, num)
                latents = cuda.to_cpu(ext.data)
                encodings[:] = cuda.to_cpu(encoding.data)

        # latents[i] = cuda.to_cpu(latent.data)
        return latents, encodings, valid_ts

    def load_model(self):
        try:
            file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.model_path)
            print(os.path.dirname(__file__))
            serializers.load_hdf5(file_path + 'enc.model', self.enc_models[0])
            serializers.load_hdf5(file_path + 'enc.model', self.enc_models[1])

            serializers.load_hdf5(file_path + 'att_enc.model', self.att_enc_models[0])
            serializers.load_hdf5(file_path + 'att_enc.model', self.att_enc_models[1])

            serializers.load_hdf5(file_path + 'att_gen.model', self.att_gen_models[0])
            serializers.load_hdf5(file_path + 'att_gen.model', self.att_gen_models[1])

        except Exception as inst:
            print(inst)
            print('cannot load the encoder model from {}'.format(file_path))

    def to_gpu(self):
            self.att_enc_models[0].to_gpu(GPU.gpus_to_use[0])
            self.att_enc_models[1].to_gpu(GPU.gpus_to_use[1])

            self.enc_models[0].to_gpu(GPU.gpus_to_use[0])
            self.enc_models[1].to_gpu(GPU.gpus_to_use[1])

            self.att_gen_models[0].to_gpu(GPU.gpus_to_use[0])
            self.att_gen_models[1].to_gpu(GPU.gpus_to_use[1])

    def read_image(self, path, noise=[0.0, 0.0]):
        image = Image.open(path)
        camera_id = 1
        image = self.transformer.apply_homography(image, camera_id, noise=noise)
        return image

    def pre_process_image(self, image):
        if self.num_channels == 1:
            image = image.convert('L')

        if self.num_channels >= 3:
            image = image.convert('RGB')
            image = np.array(image)
            image = image.transpose((2, 0, 1))
            image = image[:, :, ::-1].copy()

        image = np.asarray(image, dtype=np.float32)
        image = image / 127.5 - 1
        return image

def show_image(images):
    img = images
    img = img.transpose(1, 2, 0)
    img = (img + 1) *127.5
    img = img.astype(np.uint8)
    print(img.dtype, np.max(img), np.min(img), np.shape(img))
    img = Image.fromarray(img, "RGB")
    img.show()

if __name__ == '__main__':
    ED = EncodeDataset()
