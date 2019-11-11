import math
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import variable
from chainer import cuda, Variable
from chainer.initializers import Normal

class Encoder(chainer.Chain):
    def __init__(self, density=1, latent_size=64, text_encoding_size=128, att_size=28, word_num=10, shape_size=10, color_size=10):
        self.att_size = att_size
        self.input_size = 224
        self.density = density
        self.word_num = word_num
        super(Encoder, self).__init__(
            # LSTM part
            l1_ = L.LSTM(self.word_num, text_encoding_size),
            # a pretrained VGG feature extractor, not trained in this network
            dc1 = L.Convolution2D(3, int(16 * density), 5, stride=2, pad=2, initialW=Normal(0.02)),
            dc2 = L.Convolution2D(int(16 * density), int(32 * density), 5, stride=2, pad=2, initialW=Normal(0.02)),
            norm2 = L.BatchNormalization(int(32 * density)),
            dc2_= L.Convolution2D(int(32 * density), int(32 * density), 3, stride=1, pad=1, initialW=Normal(0.02)),
            norm2_= L.BatchNormalization(int(32 * density)),
            dc2__ = L.Convolution2D(int(32 * density), int(32 * density), 3, stride=1, pad=1, initialW=Normal(0.02)),
            norm2__ = L.BatchNormalization(int(32 * density)),
            dc3 = L.Convolution2D(int(32 * density), int(64 * density), 3, stride=2, pad=1, initialW=Normal(0.02)),
            norm3=L.BatchNormalization(int(64 * density)),
            dc3_=L.Convolution2D(int(64 * density), int(64 * density), 3, stride=1, pad=1, initialW=Normal(0.02)),
            norm3_=L.BatchNormalization(int(64 * density)),
            dc3__=L.Convolution2D(int(64 * density), int(64 * density), 3, stride=1, pad=1, initialW=Normal(0.02)),
            norm3__=L.BatchNormalization(int(64 * density)),
            dc4=L.Convolution2D(int(64 * density), int(128 * density), 3, stride=1, pad=1, initialW=Normal(0.02)),
            norm4=L.BatchNormalization(int(128 * density)),
            dc5_mean = L.Linear(self.att_size * self.att_size * int(128 * density), latent_size, initialW=Normal(0.02)),
            dc5_var = L.Linear(self.att_size * self.att_size * int(128 * density), latent_size, initialW=Normal(0.02)),
            # dc6 = L.Linear(robot_latent_size, 3, initialW=Normal(0.02)),

            fc_video0 = L.Convolution2D(int(128 * density), int(128 * density), 1, stride=1, pad=0, initialW=Normal(0.02), nobias=True),
            norm0 = L.BatchNormalization(int(128 * density), use_gamma = False),
            fc_video1 = L.Convolution2D(int(128 * density), int(8 * density), 1, stride=1, pad=0, initialW=Normal(0.02), nobias=True),
            norm1 = L.BatchNormalization(int(8 * density), use_gamma = False),
            fc_video2 = L.Convolution2D(int(8 * density), int(8 * density), 1, stride=1, pad=0, initialW=Normal(0.02)),

            # Text Input Layers
            fc_text0 = L.Linear(text_encoding_size, int(8 * density), initialW=Normal(0.02), nobias=True),
            norm_text0 = L.BatchNormalization(int(8 * density)),
            fc_text1 = L.Linear(int(8 * density), int(8 * density), initialW=Normal(0.02)),
            # norm_text1 = L.BatchNormalization(128),

            #Attention Extraction
            norm_mix = L.BatchNormalization(int(8 * density), use_gamma = False),
            fc_mix0 = L.Convolution2D(int(8 * density), 1, 1, stride=1, pad=0, initialW=Normal(0.02)),
            # fc7 = L.Linear(64 * 7 * 7, 64 * self.att_size * self.att_size, initialW=Normal(0.02)),

            #Classifier
            fc_cls0 = L.Linear(int(8 * density), int(8 * density), initialW=Normal(0.02), nobias=True),
            norm_cls0 = L.BatchNormalization(int(8 * density)),
            fc5 = L.Linear(int(8 * density), shape_size, initialW=Normal(0.02)),
            fc6 = L.Linear(int(8 * density), color_size, initialW=Normal(0.02)),

            #Latent extraction
            # norm_D = L.BatchNormalization(int((128 + 1) * density)),
            # fc_toz_mean = L.Linear(int((128 + 1) * density), latent_size, initialW=Normal(0.02)),
            # fc_toz_var = L.Linear(int((128 + 1) * density), latent_size, initialW=Normal(0.02)),
        )
    
        # with self.init_scope():
        #     self.D = variable.Parameter(Normal(0.02), (int(density), self.att_size * self.att_size))

    def __call__(self, x, sentence, att_mask=None, train=True):
        with chainer.using_config('train', train), chainer.using_config('enable_backprop', train):
            xp = cuda.get_array_module(x.data)

            h1 = F.leaky_relu(self.dc1(x))
            h2 = F.leaky_relu(self.norm2(self.dc2(h1)))
            h2_ = F.leaky_relu(self.norm2_(self.dc2_(h2)))
            h2__ = F.leaky_relu(self.norm2__(self.dc2__(h2_)))
            h3 = F.leaky_relu(self.norm3(self.dc3(h2__)))
            h3_ = F.leaky_relu(self.norm3_(self.dc3_(h3)))
            h3__ = F.leaky_relu(self.norm3__(self.dc3__(h3_)))
            h4 = F.leaky_relu(self.norm4(self.dc4(h3__)))
            mean = self.dc5_mean(h4)
            var = F.tanh(self.dc5_var(h4))
            rand = xp.random.normal(0, 1, var.data.shape).astype(np.float32)
            z = mean + F.exp(var) * Variable(rand)
            # h6 = F.leaky_relu(self.dc6(h5))

            f0 = F.tanh(self.norm0(self.fc_video0(h4)))
            f1 = F.tanh(self.norm1(self.fc_video1(f0)))
            f3 = self.fc_video2(f1)

            self.l1_.reset_state()
            for i in range(sentence.shape[1]):
                encoded = self.l1_(sentence[:, i])

            s0 = F.tanh(self.norm_text0(self.fc_text0(encoded)))
            s1 = self.fc_text1(s0)
            s2 = F.expand_dims(s1, axis=2)
            s2 = F.repeat(s2, self.att_size * self.att_size, axis=2)
            s2 = F.reshape(s2, (-1, int(8 * self.density), self.att_size, self.att_size))

            m3 = f3 + s2
            m3 = F.tanh(self.norm_mix(m3))
            m4 = F.reshape(self.fc_mix0(m3), (-1, self.att_size * self.att_size))
            # m4 = 20 * F.normalize(m4, axis=1)
            m4 = F.softmax(F.relu(m4), axis=1)

            # h0_ = F.reshape(F.max_pooling_2d(h0_, 2), (-1, 512, self.att_size * self.att_size))
            f3 = F.reshape(f3, (-1, 8 * self.density, self.att_size * self.att_size))
            # f2 = F.einsum('ijk,ik -> ij', h0_, h4)
            # features_rolled = None
            if train:
                masked = att_mask * m4
                features = F.einsum('ijk,ik -> ij', f3, masked)
                # features_rolled = F.einsum('ijk,ik -> ij', f3, xp.roll(masked.data, 1, axis=0))
            else:
                features = F.einsum('ijk,ik -> ij', f3, m4)
            # features = F.dropout(features, 0.5)
            #Classifier
            f0 = self.norm_cls0(F.leaky_relu(self.fc_cls0(features)))
            s2 = self.fc5(f0)
            c2 = self.fc6(f0)

            # h4 = F.reshape(h4, (-1, int(128 * self.density), self.att_size * self.att_size))
            # D_broad = F.broadcast_to(self.D, (h4.shape[0], self.D.shape[0], self.D.shape[1]))
            # toLatent = F.reshape(F.concat((h4, D_broad), axis=1), (-1, int((128 + 8) * self.density), self.att_size * self.att_size))
            # toLatent = self.norm_D(toLatent)

            # m4_prime = Variable(m4.data)
            # toZ = F.einsum('ijk,ik -> ij', toLatent, m4_prime)
            # mean = self.fc_toz_mean(toZ)
            # var = F.tanh(self.fc_toz_mean(toZ))
            # rand = xp.random.normal(0, 1, var.data.shape).astype(np.float32)
            # z = mean + F.exp(var) * Variable(rand)
            # return h5, z, mean, var, encoded, features, features_rolled, h6, F.reshape(m4, (-1, 1, self.att_size, self.att_size)), s2, c2
            return z, var, mean, encoded, features, F.reshape(m4, (-1, 1, self.att_size, self.att_size)), s2, c2

class Encoder_double_att(chainer.Chain):
    def __init__(self, density=1, latent_size=64, robot_latent_size=16, text_encoding_size=128, att_size=28, word_num=10, shape_size=10, color_size=10):
        self.att_size = att_size
        self.input_size = 224
        self.density = density
        self.word_num = word_num
        super(Encoder_double_att, self).__init__(
            # LSTM part
            l1_ = L.LSTM(self.word_num, text_encoding_size),
            # a pretrained VGG feature extractor, not trained in this network
            dc1 = L.Convolution2D(3, int(16 * density), 5, stride=2, pad=2, initialW=Normal(0.02)),
            dc2 = L.Convolution2D(int(16 * density), int(32 * density), 5, stride=2, pad=2, initialW=Normal(0.02)),
            norm2 = L.BatchNormalization(int(32 * density)),
            dc2_= L.Convolution2D(int(32 * density), int(32 * density), 3, stride=1, pad=1, initialW=Normal(0.02)),
            norm2_= L.BatchNormalization(int(32 * density)),
            dc2__ = L.Convolution2D(int(32 * density), int(32 * density), 3, stride=1, pad=1, initialW=Normal(0.02)),
            norm2__ = L.BatchNormalization(int(32 * density)),
            dc3 = L.Convolution2D(int(32 * density), int(64 * density), 3, stride=2, pad=1, initialW=Normal(0.02)),
            norm3=L.BatchNormalization(int(64 * density)),
            dc3_=L.Convolution2D(int(64 * density), int(64 * density), 3, stride=1, pad=1, initialW=Normal(0.02)),
            norm3_=L.BatchNormalization(int(64 * density)),
            dc3__=L.Convolution2D(int(64 * density), int(64 * density), 3, stride=1, pad=1, initialW=Normal(0.02)),
            norm3__=L.BatchNormalization(int(64 * density)),
            dc4=L.Convolution2D(int(64 * density), int(128 * density), 3, stride=1, pad=1, initialW=Normal(0.02)),
            norm4=L.BatchNormalization(int(128 * density)),
            dc5 = L.Linear(self.att_size * self.att_size * int(128 * density), 32, initialW=Normal(0.02)),
            # dc6 = L.Linear(robot_latent_size, 3, initialW=Normal(0.02)),

            fc_video0 = L.Convolution2D(int(128 * density), int(128 * density), 1, stride=1, pad=0, initialW=Normal(0.02), nobias=True),
            norm0 = L.BatchNormalization(int(128 * density), use_gamma = False),
            fc_video1 = L.Convolution2D(int(128 * density), int(16 * density), 1, stride=1, pad=0, initialW=Normal(0.02), nobias=True),
            norm1 = L.BatchNormalization(int(16 * density), use_gamma = False),
            fc_video2 = L.Convolution2D(int(16 * density), int(16 * density), 1, stride=1, pad=0, initialW=Normal(0.02)),

            # Text Input Layers
            fc_text0 = L.Linear(text_encoding_size, int(8 * density), initialW=Normal(0.02), nobias=True),
            norm_text0 = L.BatchNormalization(int(8 * density)),
            fc_text1 = L.Linear(int(8 * density), int(16 * density), initialW=Normal(0.02)),
            # norm_text1 = L.BatchNormalization(128),

            #Attention Extraction
            norm_mix = L.BatchNormalization(int(16 * density), use_gamma = False),
            fc_mix0 = L.Convolution2D(int(16 * density), 1, 1, stride=1, pad=0, initialW=Normal(0.02)),
            # fc7 = L.Linear(64 * 7 * 7, 64 * self.att_size * self.att_size, initialW=Normal(0.02)),

            #Classifier
            fc_cls0 = L.Linear(int(16 * density), int(8 * density), initialW=Normal(0.02), nobias=True),
            norm_cls0 = L.BatchNormalization(int(8 * density)),
            fc5 = L.Linear(int(8 * density), shape_size, initialW=Normal(0.02)),
            fc6 = L.Linear(int(8 * density), color_size, initialW=Normal(0.02)),
            fc7 = L.Linear(int(8 * density), 7, initialW=Normal(0.02)),

            #Latent extraction
            # norm_D = L.BatchNormalization(int((128 + 1) * density)),
            # fc_toz_mean = L.Linear(int((128 + 1) * density), latent_size, initialW=Normal(0.02)),
            # fc_toz_var = L.Linear(int((128 + 1) * density), latent_size, initialW=Normal(0.02)),
        )
    
        # with self.init_scope():
        #     self.D = variable.Parameter(Normal(0.02), (int(density), self.att_size * self.att_size))

    def __call__(self, x, sentence, att_mask=None, seq_size=1, train=True):
        with chainer.using_config('train', train), chainer.using_config('enable_backprop', train):
            xp = cuda.get_array_module(x.data)

            h1 = F.leaky_relu(self.dc1(x))
            h2 = F.leaky_relu(self.norm2(self.dc2(h1)))
            h2_ = F.leaky_relu(self.norm2_(self.dc2_(h2)))
            h2__ = F.leaky_relu(self.norm2__(self.dc2__(h2_)))
            h3 = F.leaky_relu(self.norm3(self.dc3(h2__)))
            h3_ = F.leaky_relu(self.norm3_(self.dc3_(h3)))
            h3__ = F.leaky_relu(self.norm3__(self.dc3__(h3_)))
            h4 = F.leaky_relu(self.norm4(self.dc4(h3__)))
            h5 = F.leaky_relu(self.dc5(h4))
            # h6 = F.leaky_relu(self.dc6(h5))

            f0 = F.tanh(self.norm0(self.fc_video0(h4)))
            f1 = F.tanh(self.norm1(self.fc_video1(f0)))
            f3 = self.fc_video2(f1)

            self.l1_.reset_state()
            for i in range(sentence.shape[1]):
                encoded = self.l1_(sentence[:, i])

            s0 = F.tanh(self.norm_text0(self.fc_text0(encoded)))
            s1 = self.fc_text1(s0)
            s2 = F.expand_dims(s1, axis=2)
            s2 = F.repeat(s2, self.att_size * self.att_size, axis=2)
            s2 = F.reshape(s2, (-1, int(16 * self.density), self.att_size, self.att_size))

            m3 = f3 + s2
            m3 = F.tanh(self.norm_mix(m3))
            m4 = F.reshape(self.fc_mix0(m3), (-1, self.att_size * self.att_size))
            m4 = 10 * F.normalize(m4, axis=1)
            m4 = F.softmax(m4, axis=1)

            # h0_ = F.reshape(F.max_pooling_2d(h0_, 2), (-1, 512, self.att_size * self.att_size))
            f3 = F.reshape(f3, (-1, 16 * self.density, self.att_size * self.att_size))
            # f2 = F.einsum('ijk,ik -> ij', h0_, h4)
            # features_rolled = None
            if train:
                masked = m4 * att_mask
                features = F.einsum('ijk,ik -> ij', f3, masked)
                # features_rolled = F.einsum('ijk,ik -> ij', f3, xp.roll(masked.data, 1, axis=0))
            else:
                features = F.einsum('ijk,ik -> ij', f3, m4)
            # features = F.dropout(features, 0.5)
            #Classifier
            f0 = self.norm_cls0(F.leaky_relu(self.fc_cls0(features)))
            s2 = self.fc5(f0)
            c2 = self.fc6(f0)
            j2 = self.fc7(f0)

            # h4 = F.reshape(h4, (-1, int(128 * self.density), self.att_size * self.att_size))
            # D_broad = F.broadcast_to(self.D, (h4.shape[0], self.D.shape[0], self.D.shape[1]))
            # toLatent = F.reshape(F.concat((h4, D_broad), axis=1), (-1, int((128 + 8) * self.density), self.att_size * self.att_size))
            # toLatent = self.norm_D(toLatent)

            # m4_prime = Variable(m4.data)
            # toZ = F.einsum('ijk,ik -> ij', toLatent, m4_prime)
            # mean = self.fc_toz_mean(toZ)
            # var = F.tanh(self.fc_toz_mean(toZ))
            # rand = xp.random.normal(0, 1, var.data.shape).astype(np.float32)
            # z = mean + F.exp(var) * Variable(rand)
            # return h5, z, mean, var, encoded, features, features_rolled, h6, F.reshape(m4, (-1, 1, self.att_size, self.att_size)), s2, c2
            return h5, encoded, features, F.reshape(m4, (-1, 1, self.att_size, self.att_size)), s2, c2, j2


class Generator(chainer.Chain):
    """
    This implemention is very similar to the encoder_text_tower. 
    Convolution layers has been replaced with deconvolution layers.

    This implemention receives a latent vector plus two and two one-hot vectors corresponding to factorized
    features of the main object encoded in the image. For instance, if the image contains a 
    red sphere, the inputs will <image>,"red","round". 

    Intent: The hypothesis is that by providing HLPs during training and also during testing,
    we get a better generative results _of the particular object_. 

    Validation: We can check the reconstruction error metric, but we can also check this 
    with visual inspection of the reconstructed version for different HLP inputs
    """
    def __init__(self, density=1, size=64, latent_size=100, channel=3):
        filter_size = 2
        self.intermediate_size = size // 8
        assert (size % 16 == 0)
        initial_size = size // 16
        super(Generator, self).__init__(
            g1=L.Linear(latent_size, initial_size * initial_size * int(128 * density),
                        initialW=Normal(0.02)),
            norm1=L.BatchNormalization(initial_size * initial_size * int(128 * density)),
            g2=L.Deconvolution2D(int(128 * density), int(64 * density), filter_size, stride=2, pad=0,
                                 initialW=Normal(0.02)),
            norm2=L.BatchNormalization(int(64 * density)),
            g2_=L.Deconvolution2D(int(64 * density), int(64 * density), 3, stride=1, pad=1,
                                 initialW=Normal(0.02)),
            norm2_=L.BatchNormalization(int(64 * density)),
            g2_p=L.Deconvolution2D(int(64 * density), int(64 * density), 3, stride=1, pad=1,
                                initialW=Normal(0.02)),
            norm2_p=L.BatchNormalization(int(64 * density)),
            g3=L.Deconvolution2D(int(64 * density), int(32 * density), filter_size, stride=2, pad=0,
                                 initialW=Normal(0.02)),
            norm3=L.BatchNormalization(int(32 * density)),
            g3_=L.Deconvolution2D(int(32 * density), int(32 * density), 3, stride=1, pad=1,
                                 initialW=Normal(0.02)),
            norm3_=L.BatchNormalization(int(32 * density)),
            g3_p=L.Deconvolution2D(int(32 * density), int(32 * density), 3, stride=1, pad=1,
                                initialW=Normal(0.02)),
            norm3_p=L.BatchNormalization(int(32 * density)),
            g4=L.Deconvolution2D(int(32 * density), int(16 * density), filter_size, stride=2, pad=0,
                                 initialW=Normal(0.02)),
            norm4=L.BatchNormalization(int(16 * density)),
            g4_=L.Deconvolution2D(int(16 * density), int(16 * density), 3, stride=1, pad=1,
                                 initialW=Normal(0.02)),
            norm4_=L.BatchNormalization(int(16 * density)),
            g5=L.Deconvolution2D(int(16 * density), channel, filter_size, stride=2, pad=0,
                                 initialW=Normal(0.02)),
        )
        self.density = density
        self.latent_size = latent_size
        self.initial_size = initial_size

    def __call__(self, z, train=True):
        with chainer.using_config('train', train), chainer.using_config('enable_backprop', train):
            h1 = F.reshape(F.relu(self.norm1(self.g1(z))),
                           (z.data.shape[0], int(128 * self.density), self.initial_size, self.initial_size))
            h2 = F.relu(self.norm2(self.g2(h1)))
            h2_ = F.relu(self.norm2_(self.g2_(h2)))
            h2_p = F.relu(self.norm2_p(self.g2_p(h2_)))
            h2_ = h2_ + h2_p
            h3 = F.relu(self.norm3(self.g3(h2_)))
            h3_ = F.relu(self.norm3_(self.g3_(h3)))
            h3_p = F.relu(self.norm3_p(self.g3_p(h3_)))
            h3_ = h3_ + h3_p
            h4 = F.relu(self.norm4(self.g4(h3_)))
            h4_ = F.relu(self.norm4_(self.g4_(h4)))
            return F.tanh(self.g5(h4_))

class Discriminator(chainer.Chain):
    """
    A discriminator for classifying the both the regular and masked images into their HLP groups.
    The discrimiantor can receive both regular and masked images as input and it will try to 
    classify them based on the object of interest in the image. The discriminator will either 
    mark the image as fake or it will match it to one the shapes and colors. 

    The regular and masked images have will go through shared convolution and separate FF layers at
    the end. The first 8 convoltion layers are shared and there is separated Fully-connected layers
    for regular and masked images.

    The discriminator will be used in an adversarial setup with the encoder and the generator.
    The discriminator tries to mark the images generated by the generator as fake and classify 
    real images to their correct class.

    Validation: One can check the classification error for real and fake images  
    """
    def __init__(self, density=1, size=64, channel=3):
        assert (size % 16 == 0)
        initial_size = size // 16
        super(Discriminator, self).__init__(
            dc1=L.Convolution2D(channel, int(16 * density), 4, stride=2, pad=1,
                                initialW=Normal(0.02)),
            dc2=L.Convolution2D(int(16 * density), int(32 * density), 4, stride=2, pad=1,
                                initialW=Normal(0.02)),
            norm2=L.BatchNormalization(int(32 * density)),
            # An extra layer to make the network deeper and not changing the feature sizes
            dc2_=L.Convolution2D(int(32 * density), int(32 * density), 3, stride=1, pad=1,
                                 initialW=Normal(0.02)),
            norm2_=L.BatchNormalization(int(32 * density)),
            # "plus layer" another extra layer added to make it deeper with stride = 1 but this one has 
            # a skip connection between input and output
            dc2_p=L.Convolution2D(int(32 * density), int(32 * density), 3, stride=1, pad=1,
                                initialW=Normal(0.02)),
            norm2_p=L.BatchNormalization(int(32 * density)),
            dc3=L.Convolution2D(int(32 * density), int(64 * density), 4, stride=2, pad=1,
                                initialW=Normal(0.02)),
            norm3=L.BatchNormalization(int(64 * density)),
            # An extra layer to make the network deeper and not changing the feature sizes
            dc3_=L.Convolution2D(int(64 * density), int(64 * density), 3, stride=1, pad=1,
                                 initialW=Normal(0.02)),
            norm3_=L.BatchNormalization(int(64 * density)),
            # "plus layer" another extra layer added to make it deeper with stride = 1 but this one has 
            # a skip connection between input and output
            dc3_p=L.Convolution2D(int(64 * density), int(64 * density), 3, stride=1, pad=1,
                                initialW=Normal(0.02)),
            norm3_p=L.BatchNormalization(int(64 * density)),
            dc4=L.Convolution2D(int(64 * density), int(128 * density), 4, stride=2, pad=1,
                                initialW=Normal(0.02)),
            norm4=L.BatchNormalization(int(128 * density)),
            dc5=L.Linear(initial_size * initial_size * int(128 * density), 2, initialW=Normal(0.02)),
        )

    def __call__(self, x, train=True):
        with chainer.using_config('train', train):
            h1 = F.leaky_relu(self.dc1(x))
            h2 = F.leaky_relu(self.norm2(self.dc2(h1)))
            h2_ = F.leaky_relu(self.norm2_(self.dc2_(h2)))
            h2_p = F.leaky_relu(self.norm2_p(self.dc2_p(h2_)))
            h2_ = h2_ + h2_p
            h3 = F.leaky_relu(self.norm3(self.dc3(h2_)))
            h3_ = F.leaky_relu(self.norm3_(self.dc3_(h3)))
            h3_p = F.leaky_relu(self.norm3_p(self.dc3_p(h3_)))
            h3_ = h3_ + h3_p
            h4 = F.leaky_relu(self.norm4(self.dc4(h3_)))
            return self.dc5(h4), h3