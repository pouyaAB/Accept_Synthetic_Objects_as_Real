import numpy as np
import time
import sys
import os
import copy
import math
import scipy.ndimage
import chainer.functions as F
from PIL import Image
import threading
import signal
import copy

from matplotlib.pyplot import margins

from gpu import GPU
import chainer
import chainer.distributions as D
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
import autoencoders.reduction_att
import autoencoders.tower
from evaluation import evaluation
from nf_mdn_rnn import RobotController

from DatasetController_hybrid import DatasetController

from local_config import config

locals().update(config)
def signal_handler(signal, frame):
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

class ModelController:
    """
        Handles the model interactions
    """
    def __init__(self, image_size, latent_size, batch_size, sequence_size,
                 num_channels=3, save_dir="model/", epoch_num=320,
                 sample_image_cols=5, sample_image_rows=4, load_models=True):

        self.evaluator = evaluation()
        self.dataset_ctrl = DatasetController(batch_size=int(batch_size), sequence_size=sequence_size, shuffle_att=False)
        self.num_tasks = len(config['tasks'])
        self.image_size = 128
        self.att_size = 16
        self.normer = image_size * image_size * 3 * 60
        self.vocabularySize = 34
        self.text_encoding_size = 32
        self.batch_size = batch_size
        self.sequence_size = sequence_size
        self.num_channels = num_channels
        self.latent_size = latent_size
        self.hidden_dimension = hidden_dimension
        self.num_mixture = num_mixture
        self.output_size = output_size
        self.save_dir = save_dir
        self.epoch_num = epoch_num
        self.load_models = load_models
        self.last_best_result = 100
        self.save_model_period = 500
        self.save_evaluation_metrics = 500
        self.save_sample_image_interval = 250

        self.sample_image_cols = sample_image_cols
        self.sample_image_rows = sample_image_rows

        self.generator_test = self.dataset_ctrl.get_next_batch(task=['5002', '5001'], from_start_prob=0, train=False, camera='camera-1', return_mix=True)

        self.num_descs = self.dataset_ctrl.num_all_objects_describtors
        self.num_objects = self.dataset_ctrl.num_all_objects

        images, images_mix, \
        images_single, images_single_mix, \
        images_multi, \
        _, _, sentence, \
        objects, objs_one_hot, descriptions, descs_one_hot = next(self.generator_test)
        
        images = images_single_mix[:, 0]
        images_multi = images_multi[:, 0]
        images = np.reshape(images, (-1, self.num_channels, self.image_size, self.image_size))
        images_multi = np.reshape(images_multi, (-1, self.num_channels, self.image_size, self.image_size))

        sample_size = int(self.sample_image_cols * self.sample_image_rows/2)
        temp = np.asarray(images[:sample_size], dtype=np.float32)
        self.sample_images = np.concatenate((temp, np.asarray(images_multi[:sample_size], dtype=np.float32)), axis=0)
        temp = np.asarray(sentence[:sample_size], dtype=np.float32)
        self.sample_sentence = np.concatenate((temp, temp), axis=0)
        temp = np.asarray(objs_one_hot[:sample_size], dtype=np.float32)
        self.sample_objs_one_hot = np.concatenate((temp, temp), axis=0)
        temp = np.asarray(descs_one_hot[:sample_size], dtype=np.float32)
        self.sample_descs_one_hot = np.concatenate((temp, temp), axis=0)

        print(np.squeeze(objects)[:sample_size])
        print(np.squeeze(descriptions)[:sample_size])
        
        self.generator_test = self.dataset_ctrl.get_next_batch(task=['5002', '5001'], from_start_prob=0, train=False, camera='camera-1', return_multi=False, return_mix=True)
        self.generator = self.dataset_ctrl.get_next_batch(task=['5002', '5001'], from_start_prob=0, camera='camera-1', return_single=True, return_multi=True, return_mix=True, return_dem=True)

        self.enc_model = autoencoders.reduction_att.Encoder_text_tower(density=8, size=self.image_size, latent_size=latent_size, att_size=self.att_size, num_objects=self.num_objects, num_descriptions=self.num_descs)
        self.att_enc_model = autoencoders.tower.Encoder(density=8, size=self.image_size, latent_size=latent_size, channel=self.num_channels)
        self.att_gen_model = autoencoders.reduction_att.Generator(density=8, size=image_size, latent_size=latent_size, att_size=self.att_size, channel=2 * self.num_channels, num_objects=self.num_objects, num_descriptions=self.num_descs)
        self.dis_model = autoencoders.reduction_att.Discriminator(density=8, size=self.image_size, channel=self.num_channels, num_obj=self.num_objects + 1, num_desc=self.num_descs + 1)
        self.mdn_model = RobotController(self.latent_size + self.num_objects + self.num_descs + 4, hidden_dimension, output_size, num_mixture, auto_regressive=False)

        self.enc_models = [self.enc_model]
        self.att_enc_models = [self.att_enc_model]
        self.att_gen_models = [self.att_gen_model]
        self.dis_models = [self.dis_model]
        self.mdn_models = [self.mdn_model]

        self.learning_rate = 0.00001
        self.WeightDecay = 0.00001

        self.optimizer_enc = optimizers.Adam(alpha=self.learning_rate, beta1=0.9)
        self.optimizer_enc.setup(self.enc_models[0])
        self.optimizer_enc.add_hook(chainer.optimizer.WeightDecay(self.WeightDecay))

        self.optimizer_att_enc = optimizers.Adam(alpha=self.learning_rate, beta1=0.9)
        self.optimizer_att_enc.setup(self.att_enc_models[0])
        self.optimizer_att_enc.add_hook(chainer.optimizer.WeightDecay(self.WeightDecay))

        self.optimizer_att_gen = optimizers.Adam(alpha=self.learning_rate, beta1=0.9)
        self.optimizer_att_gen.setup(self.att_gen_models[0])
        self.optimizer_att_gen.add_hook(chainer.optimizer.WeightDecay(self.WeightDecay))

        self.optimizer_dis = optimizers.Adam(alpha=self.learning_rate, beta1=0.9)
        self.optimizer_dis.setup(self.dis_models[0])
        self.optimizer_dis.add_hook(chainer.optimizer.WeightDecay(self.WeightDecay))

        self.optimizer_mdn = optimizers.Adam(alpha=self.learning_rate, beta1=0.9)
        self.optimizer_mdn.setup(self.mdn_models[0])
        self.optimizer_mdn.add_hook(chainer.optimizer.WeightDecay(0.00001))

        for i in range(GPU.num_gpus - 1):
            self.enc_models.append(copy.deepcopy(self.enc_model))
            self.att_enc_models.append(copy.deepcopy(self.att_enc_model))
            self.att_gen_models.append(copy.deepcopy(self.att_gen_model))
            self.dis_models.append(copy.deepcopy(self.dis_model))
            self.mdn_models.append(copy.deepcopy(self.mdn_model))

        self.batch_gpu_threads = [None] * GPU.num_gpus

        if self.load_models:
            self.load_model()
        self.to_gpu()

    def reset_all(self, models):
        for model in models:
            model.reset_state()

    def show_image(self, images):
        img = images
        img = img.transpose(1, 2, 0)
        img = (img + 1) * 127.5
        img = img.astype(np.uint8)
        print(img.dtype, np.max(img), np.min(img), np.shape(img))
        img = Image.fromarray(img, "RGB")
        img.show()

    def mix(self, a, b):
        sh = a.shape
        sh = (sh[0] * 2,) + a.shape[1:]
        c = np.empty(sh, dtype=a.dtype)
        c[0::2] = a
        c[1::2] = b
        return c

    def train(self):
        xp = cuda.cupy
        cuda.get_device(GPU.main_gpu).use()

        self.save_sample_images(epoch=0, batch=0)
        for epoch in range(1, self.epoch_num + 1):
            print('\n ------------- epoch {0} started ------------'.format(epoch))
            batches_passed = 0
            while batches_passed < 1000:
                import gc
                gc.collect()
                batches_passed += 1
                batch_start_time = time.time()

                images_dem, images_dem_mix, \
                images_single, images_single_mix, \
                images_multi, \
                joints, batch_one_hot, sentence, \
                objs, objs_one_hot, descs, descs_one_hot = next(self.generator)

                for k, g in enumerate(GPU.gpus_to_use):
                    self.batch_gpu_threads[k] = threading.Thread(target=self.handle_gpu_batch, args=(
                        epoch, batches_passed, batch_start_time, k, g, \
                        images_dem, images_dem_mix, \
                        images_single, images_single_mix, \
                        images_multi, \
                        joints, batch_one_hot, \
                        objs, objs_one_hot, sentence, descs, descs_one_hot))

                    self.batch_gpu_threads[k].start()

                for i in range(GPU.num_gpus):
                    self.batch_gpu_threads[i].join()

                self.add_grads()
                self.optimizer_enc.update()
                self.optimizer_att_enc.update()
                self.optimizer_att_gen.update()
                self.optimizer_dis.update()
                self.optimizer_mdn.update()
                self.copy_params()

                current_batch = batches_passed
                if current_batch % self.save_sample_image_interval == 0:
                    self.save_sample_images(epoch=epoch, batch=current_batch)

                if current_batch % self.save_model_period == self.save_model_period - 1:
                    self.save_models()

            # self.save_models()
            self.save_sample_images(epoch=epoch, batch=batches_passed)

    def handle_gpu_batch(self, epoch, batches_passed, batch_start_time, k, g, \
                            att_images, att_images_mix, \
                            att_images_multi, att_images_multi_mix, \
                            att_images_real_multi, \
                            joints, batch_one_hot, \
                            objects, objs_one_hot, sentence, descriptions, descriptions_one_hot):

        xp = cuda.cupy
        cuda.get_device(g).use()
        self.enc_models[k].cleargrads()
        self.att_enc_models[k].cleargrads()
        self.att_gen_models[k].cleargrads()
        self.dis_models[k].cleargrads()
        self.mdn_models[k].cleargrads()
        self.reset_all([self.mdn_models[k]])

        gpu_batch_size = self.batch_size // GPU.num_gpus

        att_images = att_images[k * gpu_batch_size:(k + 1) * gpu_batch_size]
        att_images_mix = att_images_mix[k * gpu_batch_size:(k + 1) * gpu_batch_size]
        att_images_multi = att_images_multi[k * gpu_batch_size:(k + 1) * gpu_batch_size]
        att_images_multi_mix = att_images_multi_mix[k * gpu_batch_size:(k + 1) * gpu_batch_size]
        att_images_real_multi = att_images_real_multi[k * gpu_batch_size:(k + 1) * gpu_batch_size]

        objects = np.asarray(objects[k * gpu_batch_size:(k + 1) * gpu_batch_size], dtype=np.int32)
        objects = np.repeat(objects[:, np.newaxis], self.sequence_size, axis=1)

        objs_one_hot = np.asarray(objs_one_hot[k * gpu_batch_size:(k + 1) * gpu_batch_size], dtype=np.float32)
        objs_one_hot = np.repeat(objs_one_hot[:, np.newaxis], self.sequence_size, axis=1)

        descriptions = np.asarray(descriptions[k * gpu_batch_size:(k + 1) * gpu_batch_size], dtype=np.int32)
        descriptions = np.repeat(descriptions[:, np.newaxis], self.sequence_size, axis=1)
        
        descriptions_one_hot = np.asarray(descriptions_one_hot[k * gpu_batch_size:(k + 1) * gpu_batch_size], dtype=np.float32)
        descriptions_one_hot = np.repeat(descriptions_one_hot[:, np.newaxis], self.sequence_size, axis=1)

        att_images = att_images.transpose(1, 0, 2, 3, 4)
        att_images_mix = att_images_mix.transpose(1, 0, 2, 3, 4)
        att_images_multi = att_images_multi.transpose(1, 0, 2, 3, 4)
        att_images_multi_mix = att_images_multi_mix.transpose(1, 0, 2, 3, 4)
        att_images_real_multi = att_images_real_multi.transpose(1, 0, 2, 3, 4)

        objects = objects.transpose(1, 0)
        objs_one_hot = objs_one_hot.transpose(1, 0, 2)
        descriptions = descriptions.transpose(1, 0)
        descriptions_one_hot = descriptions_one_hot.transpose(1, 0, 2)

        objects = np.squeeze(np.reshape(objects, (self.sequence_size * gpu_batch_size, -1)))
        objs_one_hot = np.squeeze(np.reshape(objs_one_hot, (self.sequence_size * gpu_batch_size, -1)))
        descriptions = np.squeeze(np.reshape(descriptions, (self.sequence_size * gpu_batch_size, -1)))
        descriptions_one_hot = np.squeeze(np.reshape(descriptions_one_hot, (self.sequence_size * gpu_batch_size, -1)))

        joints = joints.transpose(1, 0, 2)
        joints = np.asarray(joints[:, k * gpu_batch_size:(k + 1) * gpu_batch_size], dtype=np.float32)
        joints = Variable(cuda.to_gpu(joints, g))

        batch_one_hot = np.asarray(batch_one_hot[k * gpu_batch_size:(k + 1) * gpu_batch_size], dtype=np.float32)
        batch_one_hot = np.repeat(batch_one_hot[np.newaxis], self.sequence_size, axis=0)
        batch_one_hot = np.reshape(batch_one_hot, (self.sequence_size * gpu_batch_size, 4))
        batch_one_hot = Variable(cuda.to_gpu(batch_one_hot, g))

        att_images = np.reshape(att_images, (-1, self.num_channels, self.image_size, self.image_size))
        x_in_att = Variable(cuda.to_gpu(np.asarray(att_images, dtype=np.float32), g))
        att_images_mix = np.reshape(att_images_mix, (-1, self.num_channels, self.image_size, self.image_size))
        x_in_att_mix = Variable(cuda.to_gpu(np.asarray(att_images_mix, dtype=np.float32), g))
        att_images_multi = np.reshape(att_images_multi, (-1, self.num_channels, self.image_size, self.image_size))
        x_in_att_multi = Variable(cuda.to_gpu(np.asarray(att_images_multi, dtype=np.float32), g))
        att_images_multi_mix = np.reshape(att_images_multi_mix, (-1, self.num_channels, self.image_size, self.image_size))
        x_in_att_multi_mix = Variable(cuda.to_gpu(np.asarray(att_images_multi_mix, dtype=np.float32), g))
        att_images_real_multi = np.reshape(att_images_real_multi, (-1, self.num_channels, self.image_size, self.image_size))
        x_in_att_real_multi = Variable(cuda.to_gpu(np.asarray(att_images_real_multi, dtype=np.float32), g))

        objects_var = Variable(cuda.to_gpu(objects, g))
        desc_var = Variable(cuda.to_gpu(descriptions, g))
        objects_hot_var = Variable(cuda.to_gpu(objs_one_hot, g))
        desc_hot_var = Variable(cuda.to_gpu(descriptions_one_hot, g))
        
        att0, s0, c0 = self.enc_models[k](x_in_att, objects_hot_var, desc_hot_var, train=True)
        m_att0, m_s0, m_c0 = self.enc_models[k](x_in_att_mix, objects_hot_var, desc_hot_var, train=True)
        att00, s00, c00 = self.enc_models[k](x_in_att_multi, objects_hot_var, desc_hot_var, train=True)
        m_att00, m_s00, m_c00 = self.enc_models[k](x_in_att_multi_mix, objects_hot_var, desc_hot_var, train=True)
        real_att0, real_s0, real_c0 = self.enc_models[k](x_in_att_real_multi, objects_hot_var, desc_hot_var, train=True)

        l1_norm_att =  F.sum(att0)
        l1_norm_att += F.sum(m_att0)
        l1_norm_att += F.sum(att00)
        l1_norm_att += F.sum(m_att00)
        l1_norm_att += F.sum(real_att0)
        l1_norm_att /= 5 * gpu_batch_size * self.sequence_size * self.att_size * self.att_size

        # att0 = F.normalize(att0, axis=1)
        att0 = F.reshape(att0, (-1, 1, self.att_size, self.att_size))
        att0 = F.resize_images(att0, (self.image_size, self.image_size))
        # m_att0 = F.normalize(m_att0, axis=1)
        m_att0 = F.reshape(m_att0, (-1, 1, self.att_size, self.att_size))
        m_att0 = F.resize_images(m_att0, (self.image_size, self.image_size))
        # att00 = F.normalize(att00, axis=1)
        att00 = F.reshape(att00, (-1, 1, self.att_size, self.att_size))
        att00 = F.resize_images(att00, (self.image_size, self.image_size))
        # m_att00 = F.normalize(m_att00, axis=1)
        m_att00 = F.reshape(m_att00, (-1, 1, self.att_size, self.att_size))
        m_att00 = F.resize_images(m_att00, (self.image_size, self.image_size))
        # real_att0 = F.normalize(real_att0, axis=1)
        real_att0 = F.reshape(real_att0, (-1, 1, self.att_size, self.att_size))
        real_att0 = F.resize_images(real_att0, (self.image_size, self.image_size))

        att_classification = F.softmax_cross_entropy(s0, objects_var) + F.softmax_cross_entropy(c0, desc_var)
        att_classification += F.softmax_cross_entropy(m_s0, objects_var) + F.softmax_cross_entropy(m_c0, desc_var)
        att_classification += F.softmax_cross_entropy(s00, objects_var) + F.softmax_cross_entropy(c00, desc_var)
        att_classification += F.softmax_cross_entropy(m_s00, objects_var) + F.softmax_cross_entropy(m_c00, desc_var)
        att_classification += F.softmax_cross_entropy(real_s0, objects_var) + F.softmax_cross_entropy(real_c0, desc_var)
        att_classification /= 10

        g1 = x_in_att * att0
        g2 = x_in_att_mix * m_att0
        g3 = x_in_att_multi * att00
        g4 = x_in_att_multi_mix * m_att00
        g5 = x_in_att_real_multi * real_att0

        att_similarity = F.mean_squared_error(g1, g2)
        att_similarity += F.mean_squared_error(g3, g4)

        cir_z, cir_mean, cir_var, _ = self.att_enc_models[k](g1, train=True)
        cir_z_m, cir_mean_m, cir_var_m, _ = self.att_enc_models[k](g2, train=True)
        cir_z0, cir_mean0, cir_var0, _ = self.att_enc_models[k](g3, train=True)
        cir_z0_m, cir_mean0_m, cir_var0_m, _ = self.att_enc_models[k](g4, train=True)
        cir_z_real, cir_mean_real, cir_var_real, _ = self.att_enc_models[k](g5, train=True)

        l_prior = F.gaussian_kl_divergence(cir_mean, cir_var) / (5 * self.normer)
        l_prior += F.gaussian_kl_divergence(cir_mean_m, cir_var_m) / (5 * self.normer)
        l_prior += F.gaussian_kl_divergence(cir_mean0, cir_var0) / (5 * self.normer)
        l_prior += F.gaussian_kl_divergence(cir_mean0_m, cir_var0_m) / (5 * self.normer)
        l_prior += F.gaussian_kl_divergence(cir_mean_real, cir_var_real) / (5 * self.normer)
        l_prior /= 5

        cir_x0 = self.att_gen_models[k](cir_z, train=True)
        cir_m_x0 = self.att_gen_models[k](cir_z_m, train=True)
        cir_x00 = self.att_gen_models[k](cir_z0, train=True)
        cir_m_x00 = self.att_gen_models[k](cir_z0_m, train=True)
        cir_real_x0 = self.att_gen_models[k](cir_z_real, train=True)

        reconstruction_loss = F.mean_squared_error(x_in_att, cir_x0[:, :3]) + F.mean_squared_error(x_in_att, cir_m_x0[:, :3])
        reconstruction_loss += F.mean_squared_error(x_in_att_multi, cir_x00[:, :3]) + F.mean_squared_error(x_in_att_multi, cir_m_x00[:, :3])

        reconstruction_loss_att = F.mean_squared_error(g1, cir_x0[:, 3:]) + F.mean_squared_error(g2, cir_m_x0[:, 3:])
        reconstruction_loss_att += F.mean_squared_error(g3, cir_x00[:, 3:]) + F.mean_squared_error(g4, cir_m_x00[:, 3:])

        reconstruction_loss /= 4
        reconstruction_loss_att /= 4
        reconstruction_loss_att *= 100

        s3, c3, l3 = self.dis_models[k](cir_x0[:, :3], train=True)
        m_s3, m_c3, m_l3 = self.dis_models[k](cir_m_x0[:, :3], train=True)
        s30, c30, l30 = self.dis_models[k](cir_x00[:, :3], train=True)
        m_s30, m_c30, m_l30 = self.dis_models[k](cir_m_x00[:, :3], train=True)
        m_s30_real, m_c30_real, m_l30_real = self.dis_models[k](cir_real_x0[:, :3], train=True)

        l_dis_rec_3 = F.softmax_cross_entropy(s3, Variable(cuda.to_gpu(xp.zeros(gpu_batch_size * self.sequence_size).astype(np.int32), g)))
        m_l_dis_rec_3 = F.softmax_cross_entropy(m_s3, Variable(cuda.to_gpu(xp.zeros(gpu_batch_size * self.sequence_size).astype(np.int32), g)))
        l_dis_rec3 = F.softmax_cross_entropy(s30, Variable(cuda.to_gpu(xp.zeros(gpu_batch_size * self.sequence_size).astype(np.int32), g)))
        m_l_dis_rec3 = F.softmax_cross_entropy(m_s30, Variable(cuda.to_gpu(xp.zeros(gpu_batch_size * self.sequence_size).astype(np.int32), g)))
        real_l_dis_rec3 = F.softmax_cross_entropy(m_s30_real, Variable(cuda.to_gpu(xp.zeros(gpu_batch_size * self.sequence_size).astype(np.int32), g)))

        l_dis_rec_3 += F.softmax_cross_entropy(c3, Variable(cuda.to_gpu(xp.zeros(gpu_batch_size * self.sequence_size).astype(np.int32), g)))
        m_l_dis_rec_3 += F.softmax_cross_entropy(m_c3, Variable(cuda.to_gpu(xp.zeros(gpu_batch_size * self.sequence_size).astype(np.int32), g)))
        l_dis_rec3 += F.softmax_cross_entropy(c30, Variable(cuda.to_gpu(xp.zeros(gpu_batch_size * self.sequence_size).astype(np.int32), g)))
        m_l_dis_rec3 += F.softmax_cross_entropy(m_c30, Variable(cuda.to_gpu(xp.zeros(gpu_batch_size * self.sequence_size).astype(np.int32), g)))
        real_l_dis_rec3 += F.softmax_cross_entropy(m_c30_real, Variable(cuda.to_gpu(xp.zeros(gpu_batch_size * self.sequence_size).astype(np.int32), g)))

        l_dis_fake = (l_dis_rec_3 + m_l_dis_rec_3 + l_dis_rec3 + m_l_dis_rec3 + real_l_dis_rec3) / 10

        s2, c2, l2 = self.dis_models[k](x_in_att, train=True)
        s22, c22, l22 = self.dis_models[k](x_in_att_multi, train=True)

        l_dis_real = F.softmax_cross_entropy(s2, objects_var) 
        l_dis_real += F.softmax_cross_entropy(s22, objects_var)
        l_dis_real += F.softmax_cross_entropy(c2, desc_var)
        l_dis_real += F.softmax_cross_entropy(c22, desc_var)
        l_dis_real /= 4

        l_feature_similarity = F.mean_squared_error(l3, l2) + F.mean_squared_error(m_l3, l2)
        l_feature_similarity += F.mean_squared_error(l30, l22) + F.mean_squared_error(m_l30, l22)
        l_feature_similarity /= 8

        text_encoding = F.concat((batch_one_hot, objects_hot_var, desc_hot_var), axis=-1)
        text_encoding = F.reshape(text_encoding, (self.sequence_size, gpu_batch_size, -1))
        z_seq = F.reshape(cir_z, (self.sequence_size, gpu_batch_size, self.latent_size))
        z_seq_mix = F.reshape(cir_z_m, (self.sequence_size, gpu_batch_size, self.latent_size))
        mdn_loss, _ = self.mdn_models[k](task_encoding=text_encoding[0], image_encoding=z_seq[:-1], data_out=joints[1:], return_sample=False)
        mdn_loss_mix, _ = self.mdn_models[k](task_encoding=text_encoding[0], image_encoding=z_seq_mix[:-1], data_out=joints[1:], return_sample=False)
        robot_loss = (mdn_loss + mdn_loss_mix) / 2

        dis_loss = (l_dis_fake + 10 * l_dis_real) / (gpu_batch_size * self.sequence_size)
        loss_classifier = att_classification
        loss_enc = 10 * l_prior + 10 * l_feature_similarity + 10 * att_similarity + 2 * l1_norm_att
        loss_gen = 2 * l_feature_similarity + 200 * reconstruction_loss - dis_loss
        loss_dis = dis_loss

        self.enc_models[k].cleargrads()
        self.att_enc_models[k].cleargrads()
        self.att_gen_models[k].cleargrads()
        self.mdn_models[k].cleargrads()
        loss_net = loss_enc + loss_gen + loss_classifier + robot_loss/5
        loss_net.backward()


        g1.unchain_backward()
        g2.unchain_backward()
        g3.unchain_backward()
        g4.unchain_backward()
        g5.unchain_backward()
        reconstruction_loss_att.backward()

        cir_x0.unchain_backward()
        cir_m_x0.unchain_backward()
        cir_x00.unchain_backward()
        cir_m_x00.unchain_backward()
        cir_real_x0.unchain_backward()

        self.dis_models[k].cleargrads()
        loss_dis.backward()

        sys.stdout.write('\r' + str(batches_passed) + '/' + str(1000) +
                         ' time: {0:0.2f}, enc:{1:0.4f}, gen:{2:0.4f}, dis:{3:0.4f}, l_prior:{4:0.4f}, fea:{5:0.4f}, att_sim:{6:0.4f}, rec:{7:0.4f}, att_rec:{8:0.4f}, att_class:{9:0.4f}, norm:{10:0.4f}, mdn_loss:{11:0.4f}'.format(
                             time.time() - batch_start_time,
                             float(loss_enc.data),
                             float(loss_gen.data),
                             float(loss_dis.data),
                             float(l_prior.data),
                             float(l_feature_similarity.data),
                             float(att_similarity.data),
                             float(reconstruction_loss.data),
                             float(reconstruction_loss_att.data),
                             float(att_classification.data),
                             float(l1_norm_att.data),
                             float(robot_loss.data)
                         ))
        sys.stdout.flush()  # important

    def copy_params(self):
        for i in range(1, GPU.num_gpus):
            self.enc_models[i].copyparams(self.enc_models[0])
            self.att_enc_models[i].copyparams(self.att_enc_models[0])
            self.att_gen_models[i].copyparams(self.att_gen_models[0])
            self.dis_models[i].copyparams(self.dis_models[0])
            self.mdn_models[i].copyparams(self.mdn_models[0])

    def add_grads(self):
        for j in range(1, GPU.num_gpus):
            self.enc_models[0].addgrads(self.enc_models[j])
            self.att_enc_models[0].addgrads(self.att_enc_models[j])
            self.att_gen_models[0].addgrads(self.att_gen_models[j])
            self.dis_models[0].addgrads(self.dis_models[j])
            self.mdn_models[0].addgrads(self.mdn_models[j])

    def to_gpu(self):
        for i in range(GPU.num_gpus):
            self.enc_models[i].to_gpu(GPU.gpus_to_use[i])
            self.att_enc_models[i].to_gpu(GPU.gpus_to_use[i])
            self.att_gen_models[i].to_gpu(GPU.gpus_to_use[i])
            self.dis_models[i].to_gpu(GPU.gpus_to_use[i])
            self.mdn_models[i].to_gpu(GPU.gpus_to_use[i])

    def save_models(self):
        xp = cuda.cupy
        test_cost = self.test_testset()
        print('\nsaving the model with the test cost: ' + str(test_cost))

        serializers.save_hdf5('{0}enc.model'.format(self.save_dir), self.enc_models[0])
        serializers.save_hdf5('{0}enc.state'.format(self.save_dir), self.optimizer_enc)

        serializers.save_hdf5('{0}att_enc.model'.format(self.save_dir), self.att_enc_models[0])
        serializers.save_hdf5('{0}att_enc.state'.format(self.save_dir), self.optimizer_att_enc)

        serializers.save_hdf5('{0}att_gen.model'.format(self.save_dir), self.att_gen_models[0])
        serializers.save_hdf5('{0}att_gen.state'.format(self.save_dir), self.optimizer_att_gen)

        serializers.save_hdf5('{0}dis.model'.format(self.save_dir), self.dis_models[0])
        serializers.save_hdf5('{0}dis.state'.format(self.save_dir), self.optimizer_dis)

        serializers.save_hdf5('{0}rnn_mdn_e2e.model'.format(self.save_dir), self.mdn_models[0])
        serializers.save_hdf5('{0}rnn_mdn_e2e.state'.format(self.save_dir), self.optimizer_mdn)

        sys.stdout.flush()

    def test_testset(self):
        xp = cuda.cupy
        test_rec_loss = 0
        num_batches = 50
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):

            gpu_batch_size = self.batch_size // GPU.num_gpus

            for i in range(num_batches):
                self.reset_all([self.mdn_models[0]])
                _, _, images_single, images_single_mix, _, _, _, _, _, objs_one_hot, _, descs_one_hot = next(self.generator_test)

                images_single = np.asarray(images_single, dtype=np.float32)
                images_single = images_single.transpose(1, 0, 2, 3, 4)
                images_single_mix = np.asarray(images_single_mix, dtype=np.float32)
                images_single_mix = images_single_mix.transpose(1, 0, 2, 3, 4)
                cuda.get_device(GPU.main_gpu).use()
                
                img_input_batch_for_gpu = images_single[:, :gpu_batch_size]
                img_input_batch_for_gpu = np.reshape(img_input_batch_for_gpu,
                                                        (-1, self.num_channels, self.image_size, self.image_size))
                img_multi_input_batch_for_gpu = images_single_mix[:, :gpu_batch_size]
                img_multi_input_batch_for_gpu = np.reshape(img_multi_input_batch_for_gpu,
                                                        (-1, self.num_channels, self.image_size, self.image_size))

                objs_one_hot = np.repeat(objs_one_hot[np.newaxis], self.sequence_size, axis=0)
                obj_hot_batch_for_gpu = np.asarray(objs_one_hot[:, 0:gpu_batch_size], dtype=np.float32)
                obj_hot_batch_for_gpu = np.reshape(obj_hot_batch_for_gpu, (self.sequence_size * gpu_batch_size, -1))

                descs_one_hot = np.repeat(descs_one_hot[np.newaxis], self.sequence_size, axis=0)
                descs_hot_batch_for_gpu = np.asarray(descs_one_hot[:, 0:gpu_batch_size], dtype=np.float32)
                descs_hot_batch_for_gpu = np.reshape(descs_hot_batch_for_gpu, (self.sequence_size * gpu_batch_size, -1))

                x_in = cuda.to_gpu(img_input_batch_for_gpu, GPU.main_gpu)
                x_in_multi = cuda.to_gpu(img_multi_input_batch_for_gpu, GPU.main_gpu)
                obj_hot_batch_for_gpu = cuda.to_gpu(obj_hot_batch_for_gpu, GPU.main_gpu)
                descs_hot_batch_for_gpu = cuda.to_gpu(descs_hot_batch_for_gpu, GPU.main_gpu)

                att, _, _ = self.enc_models[0](Variable(x_in), Variable(obj_hot_batch_for_gpu), Variable(descs_hot_batch_for_gpu), train=False)
                att0, _, _ = self.enc_models[0](Variable(x_in_multi), Variable(obj_hot_batch_for_gpu), Variable(descs_hot_batch_for_gpu), train=False)

                att = F.reshape(att, (-1, 1, self.att_size, self.att_size))
                att = F.resize_images(att, (self.image_size, self.image_size)) * Variable(x_in)

                att0 = F.reshape(att0, (-1, 1, self.att_size, self.att_size))
                att0 = F.resize_images(att0, (self.image_size, self.image_size)) * Variable(x_in_multi)

                z2, mean2, var2, _ = self.att_enc_models[0](att, train=False)
                z2_m, mean2_m, var2_m, _ = self.att_enc_models[0](att0, train=False)

                x2 = self.att_gen_models[0](z2, train=False)
                x22 = self.att_gen_models[0](z2_m, train=False)


                rec_loss = F.mean_squared_error(Variable(x_in), x2[:, :3]) + F.mean_squared_error(Variable(x_in), x22[:, :3])
                rec_loss += F.mean_squared_error(att, x2[:, 3:]) + F.mean_squared_error(att0, x22[:, 3:])

                test_rec_loss += float(rec_loss.data)
        test_rec_loss = test_rec_loss / num_batches
        print('\nrecent test cost: ' + str(test_rec_loss))
        return test_rec_loss

    def save_image(self, data, filename):
        image = ((data + 1) * 128).clip(0, 255).astype(np.uint8)
        image = image[:self.sample_image_rows * self.sample_image_cols]
        image = image.reshape(
            (self.sample_image_rows, self.sample_image_cols, self.num_channels, self.image_size,
                self.image_size)).transpose(
            (0, 3, 1, 4, 2)).reshape(
            (self.sample_image_rows * self.image_size, self.sample_image_cols * self.image_size, self.num_channels))
        if self.num_channels == 1:
            image = image.reshape(self.sample_image_rows * self.image_size,
                                    self.sample_image_cols * self.image_size)
        Image.fromarray(image).save(filename)
    
    def save_image_att(self, data, att, filename):
        image = ((data + 1) * 128).clip(0, 255)
        att = np.ceil(att)
        image = (image * att).astype(np.uint8)
        image = image[:self.sample_image_rows * self.sample_image_cols]
        image = image.reshape(
            (self.sample_image_rows, self.sample_image_cols, self.num_channels, self.image_size,
                self.image_size)).transpose(
            (0, 3, 1, 4, 2)).reshape(
            (self.sample_image_rows * self.image_size, self.sample_image_cols * self.image_size, self.num_channels))
        if self.num_channels == 1:
            image = image.reshape(self.sample_image_rows * self.image_size,
                                    self.sample_image_cols * self.image_size)
        Image.fromarray(image).save(filename)

    def save_sample_images(self, epoch, batch):

        xp = cuda.cupy
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):

            images = self.sample_images
            images_bounded = self.sample_images
            cuda.get_device(GPU.main_gpu).use()
            objs_one_hot = cuda.to_gpu(self.sample_objs_one_hot, GPU.main_gpu)
            descs_one_hot = cuda.to_gpu(self.sample_descs_one_hot, GPU.main_gpu)
            
            att, _, _ = self.enc_models[0](Variable(cuda.to_gpu(images, GPU.main_gpu)), Variable(cuda.to_gpu(objs_one_hot, GPU.main_gpu)), Variable(cuda.to_gpu(descs_one_hot, GPU.main_gpu)), train=False)
            
            att_g = F.reshape(att, (-1, 1, self.att_size, self.att_size))
            att_g = F.resize_images(att_g, (self.image_size, self.image_size))
            im_att_g = att_g * Variable(cuda.to_gpu(images, GPU.main_gpu))

            z2, mean2, var2, _ = self.att_enc_models[0](im_att_g, train=False)
            x2 = self.att_gen_models[0](z2, train=False)

            att = F.reshape(att, (-1, 1, self.att_size, self.att_size))
            att = F.resize_images(att, (self.image_size, self.image_size))

            x2 = cuda.to_cpu(x2.data)
            att_g = cuda.to_cpu(att_g.data)
            att = cuda.to_cpu(att.data)

            self.save_image(x2[:, :3], 'sample/{0:03d}_{1:07d}_rec_rec.png'.format(epoch, batch))
            self.save_image(x2[:, 3:], 'sample/{0:03d}_{1:07d}_rec_att.png'.format(epoch, batch))
            self.save_image(att * images, 'sample/{0:03d}_{1:07d}_att_grey.png'.format(epoch, batch))
            self.save_image_att(images, att, 'sample/{0:03d}_{1:07d}_att_generator.png'.format(epoch, batch))
            if batch == 0:
                self.save_image(images_bounded, 'sample/org_pic.png')

    """
        Load the saved model and optimizer
    """

    def load_model(self):
        try:
            file_path = os.path.join(os.path.dirname(__file__), self.save_dir)

            serializers.load_hdf5(file_path + 'enc.model', self.enc_model)
            serializers.load_hdf5(file_path + 'enc.state', self.optimizer_enc)
            
            serializers.load_hdf5(file_path + 'att_enc.model', self.att_enc_model)
            serializers.load_hdf5(file_path + 'att_enc.state', self.optimizer_att_enc)

            serializers.load_hdf5(file_path + 'att_gen.model', self.att_gen_model)
            serializers.load_hdf5(file_path + 'att_gen.state', self.optimizer_att_gen)

            serializers.load_hdf5(file_path + 'dis.model', self.dis_model)
            serializers.load_hdf5(file_path + 'dis.state', self.optimizer_dis)

            serializers.load_hdf5(file_path + 'rnn_mdn_e2e.model', self.mdn_model)
            serializers.load_hdf5(file_path + 'rnn_mdn_e2e.state', self.optimizer_mdn)

        except Exception as inst:
            print(inst)
            print('cannot load model from {}'.format(file_path))


        self.enc_models = [self.enc_model]
        self.att_enc_models = [self.att_enc_model]
        self.att_gen_models = [self.att_gen_model]
        self.dis_models = [self.dis_model]
        self.mdn_models = [self.mdn_model]

        for i in range(GPU.num_gpus - 1):
            self.enc_models.append(copy.deepcopy(self.enc_model))
            self.att_enc_models.append(copy.deepcopy(self.att_enc_model))
            self.att_gen_models.append(copy.deepcopy(self.att_gen_model))
            self.dis_models.append(copy.deepcopy(self.dis_model))
            self.mdn_models.append(copy.deepcopy(self.mdn_model))

def show_image(images):
    # for i in range(images.shape[0]):
    img = images
    img = img.transpose(1, 2, 0)
    img = (img + 1) *127.5
    img = img.astype(np.uint8)
    print(img.dtype, np.max(img), np.min(img), np.shape(img))
    img = Image.fromarray(img, "RGB")
    img.show()
