import math
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import variable
from chainer import cuda, Variable
from chainer.initializers import Normal

class att_classifier(chainer.Chain):
    def __init__(self, density=8, num_objects=10, num_descriptions=10):
        super(att_classifier, self).__init__(
            #Classifier
            fc_cls0 = L.Linear(int(8 * density), int(8 * density), initialW=Normal(0.02), nobias=True),
            norm_cls0 = L.BatchNormalization(int(8 * density)),
            fc5 = L.Linear(int(8 * density), num_objects + 1, initialW=Normal(0.02)),
            fc6 = L.Linear(int(8 * density), num_descriptions + 1, initialW=Normal(0.02)),
        )

    def __call__(self, att, features, train=True):
        with chainer.using_config('train', train), chainer.using_config('enable_backprop', train):
            #Classifier
            features = F.einsum('ijk,ik -> ij', features, att)
            f0 = self.norm_cls0(F.leaky_relu(self.fc_cls0(features)))
            s2 = self.fc5(f0)
            c2 = self.fc6(f0)

            return s2, c2


class Encoder_text_tower(chainer.Chain):
    """
    An implementation of the "Tower" model of the VAE encoder from the paper 
    'Neural scene representation and rendering, by S. M. Ali Eslami and others at DeepMind.
    The exact numbers of the layers and were changed.

    It is basically a cVAE with multi-dimensional conditions.

    v - human level properties HLP, human classification ???? find a good name

    This system takes as input an image and two one-hot vectors corresponding to factorized
    features of the main object encoded in the image. For instance, if the image contains a 
    red sphere, the inputs will <image>,"red","round". 

    Intent: The hypothesis is that by providing HLPs during training and also during testing,
    we get a better encoding _of the particular object_. 

    Validation: We can check the reconstruction error metric, but we can also check this 
    with visual inspection of the reconstructed version for different HLP inputs
    """
    def __init__(self, density=1, size=64, latent_size=100, channel=3, att_size=32, hidden_dim=100, num_objects=10, num_descriptions=10):
        """
        density - a scaling factor for the number of channels in the convolutional layers. It is multiplied by at least
        16,32,64 and 128 as we go deeper. 
        Use: using density=8 when training the VAE separately. using density=4 when training end to end
        Intent: increase the number of features in the convolutional layers. 
        """
        assert (size % 16 == 0)
        self.att_size = att_size
        self.density = density
        self.second_size = size // 4
        initial_size = size // 16
        super(Encoder_text_tower, self).__init__(
            toConv=L.Linear(num_objects + num_descriptions, self.second_size * self.second_size * 7, initialW=Normal(0.02)),

            dc1=L.Convolution2D(channel, int(16 * density), 3, stride=2, pad=1,
                                initialW=Normal(0.02)),
            dc2=L.Convolution2D(int(16 * density), int(32 * density), 3, stride=2, pad=1,
                                initialW=Normal(0.02)),
            dc1_=L.Convolution2D(int(16 * density), int(16 * density), 3, stride=1, pad=1,
                                initialW=Normal(0.02)),
            dc2_=L.Convolution2D(int(32 * density), int(32 * density), 3, stride=1, pad=1,
                                initialW=Normal(0.02)),
            norm2=L.BatchNormalization(int(32 * density)),
            norm2_=L.BatchNormalization(int(32 * density)),
            dc2_p=L.Convolution2D(int(32 * density), int(32 * density), 3, stride=1, pad=1,
                                initialW=Normal(0.02)),
            norm2_p=L.BatchNormalization(int(32 * density)),
            dc3=L.Convolution2D(int(32 * density), int(64 * density), 3, stride=2, pad=1,
                                initialW=Normal(0.02)),
            norm3=L.BatchNormalization(int(64 * density)),
            dc3_=L.Convolution2D(int(64 * density), int(64 * density), 3, stride=1, pad=1,
                                initialW=Normal(0.02)),
            norm3_=L.BatchNormalization(int(64 * density)),
            dc3_p=L.Convolution2D(int(64 * density), int(64 * density), 3, stride=1, pad=1,
                                initialW=Normal(0.02)),
            norm3_p=L.BatchNormalization(int(64 * density)),
            dc4=L.Convolution2D(int(64 * density), int(128 * density), 3, stride=1, pad=1,
                                initialW=Normal(0.02)),
            norm4=L.BatchNormalization(int(128 * density)),
            norm_toConv=L.BatchNormalization(7),

            fc_video0 = L.Convolution2D(int(128 * density), int(128 * density), 1, stride=1, pad=0, initialW=Normal(0.02), nobias=True),
            norm0 = L.BatchNormalization(int(128 * density), use_gamma = False),
            fc_video1 = L.Convolution2D(int(128 * density), int(8 * density), 1, stride=1, pad=0, initialW=Normal(0.02), nobias=True),
            norm1 = L.BatchNormalization(int(8 * density), use_gamma = False),
            fc_video2 = L.Convolution2D(int(8 * density), int(8 * density), 1, stride=1, pad=0, initialW=Normal(0.02)),

            # Text Input Layers
            fc_text0 = L.Linear(num_objects + num_descriptions, int(8 * density), initialW=Normal(0.02), nobias=True),
            norm_text0 = L.BatchNormalization(int(8 * density)),
            fc_text1 = L.Linear(int(8 * density), int(8 * density), initialW=Normal(0.02)),

            #Attention Extraction
            norm_mix = L.BatchNormalization(int(8 * density), use_gamma = False),
            fc_mix0 = L.Convolution2D(int(8 * density), 1, 1, stride=1, pad=0, initialW=Normal(0.02)),

            #Classifier
            fc_cls0 = L.Linear(int(128 * density), int(32 * density), initialW=Normal(0.02), nobias=True),
            norm_cls0 = L.BatchNormalization(int(32 * density)),
            fc5 = L.Linear(int(32 * density), num_objects + 1, initialW=Normal(0.02)),
            fc6 = L.Linear(int(32 * density), num_descriptions + 1, initialW=Normal(0.02)),
        )

    def __call__(self, x, objects_one_hot, descs_one_hot, train=True):
        with chainer.using_config('train', train), chainer.using_config('enable_backprop', train):
            xp = cuda.get_array_module(x.data)
            h1 = F.leaky_relu(self.dc1(x))
            h1_ = F.leaky_relu(self.dc1_(h1))
            h2 = F.leaky_relu(self.norm2(self.dc2(h1_)))
            h2_ = F.leaky_relu(self.norm2_(self.dc2_(h2)))
            h2_p = F.leaky_relu(self.norm2_p(self.dc2_p(h2_)))
            h2_pp = h2_ + h2_p
            # h2_pp = F.concat((h2_pp, h0), axis=1)
            h3 = F.leaky_relu(self.norm3(self.dc3(h2_pp)))
            h3_ = F.leaky_relu(self.norm3_(self.dc3_(h3)))
            h3_p = F.leaky_relu(self.norm3_p(self.dc3_p(h3)))
            h3_ = h3_ + h3_p
            h4 = F.leaky_relu(self.norm4(self.dc4(h3_)))

            f0 = F.leaky_relu(self.norm0(self.fc_video0(h4)))
            f1 = F.leaky_relu(self.norm1(self.fc_video1(f0)))
            f3 = self.fc_video2(f1)

            s0 = F.leaky_relu(self.norm_text0(self.fc_text0(F.concat((objects_one_hot, descs_one_hot), axis=-1))))
            s1 = self.fc_text1(s0)
            s2 = F.expand_dims(s1, axis=2)
            s2 = F.repeat(s2, self.att_size * self.att_size, axis=2)
            s2 = F.reshape(s2, (-1, int(8 * self.density), self.att_size, self.att_size))

            m3 = f3 + s2
            m3 = F.tanh(self.norm_mix(m3))
            m4 = F.reshape(self.fc_mix0(m3), (-1, self.att_size * self.att_size))
            m4 = F.sigmoid(m4)
            m4 = 100 * F.normalize(F.relu(m4 - 0.5), axis=1)
            m4 = F.clip(m4, 0.0, 1.0)

            h4 = F.reshape(h4, (-1, int(self.density * 128), self.att_size * self.att_size))

            if train:
                features = F.einsum('ijk,ik -> ij', h4, m4)
            else:
                features = F.einsum('ijk,ik -> ij', h4, m4)
            #Classifier
            t0 = self.norm_cls0(F.leaky_relu(self.fc_cls0(features)))
            s2 = self.fc5(t0)
            c2 = self.fc6(t0)

            return m4, s2, c2

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
    def __init__(self, density=1, size=64, latent_size=100, att_size=8, channel=3, num_objects=8, num_descriptions=5):
        filter_size = 2
        self.density = density
        self.att_size = att_size
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

    def __call__(self, z, att_mask=None, train=True):
        with chainer.using_config('train', train), chainer.using_config('enable_backprop', train):
            z = F.normalize(z, axis=1)
            h1 = F.reshape(F.relu(self.norm1(self.g1(z))),
                           (z.data.shape[0], int(128 * self.density), self.initial_size, self.initial_size))
            # h1_att = F.relu(self.g1_att(z))
            h2 = F.relu(self.norm2(self.g2(h1)))
            h2_ = F.relu(self.norm2_(self.g2_(h2)))
            h2_ = h2_ + h2
            h2_p = F.relu(self.norm2_p(self.g2_p(h2_)))
            h2_p = h2_ + h2_p
            h3 = F.relu(self.norm3(self.g3(h2_p)))
            h3_ = F.relu(self.norm3_(self.g3_(h3)))
            h3_ = h3_ + h3
            h3_p = F.relu(self.norm3_p(self.g3_p(h3_)))
            h3_pp = h3_ + h3_p
            h4 = F.relu(self.norm4(self.g4(h3_pp)))
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
    def __init__(self, density=1, size=64, num_obj=8, num_desc=5, channel=3):
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
            dc5=L.Linear(initial_size * initial_size * int(128 * density), num_obj, initialW=Normal(0.02)),
            dc6=L.Linear(initial_size * initial_size * int(128 * density), num_desc, initialW=Normal(0.02)),
        )

    def __call__(self, x, train=True):
        with chainer.using_config('train', train):
            h1 = F.leaky_relu(self.dc1(x))
            h2 = F.leaky_relu(self.norm2(self.dc2(h1)))
            h2_ = F.leaky_relu(self.norm2_(self.dc2_(h2)))
            h2_ = h2_ + h2
            h2_p = F.leaky_relu(self.norm2_p(self.dc2_p(h2_)))
            h2_p = h2_ + h2_p
            h3 = F.leaky_relu(self.norm3(self.dc3(h2_p)))
            h3_ = F.leaky_relu(self.norm3_(self.dc3_(h3)))
            h3_ = h3_ + h3
            h3_p = F.leaky_relu(self.norm3_p(self.dc3_p(h3_)))
            h3_p = h3_ + h3_p
            h4 = F.leaky_relu(self.norm4(self.dc4(h3_p)))
            return self.dc5(h4), self.dc6(h4), h3

