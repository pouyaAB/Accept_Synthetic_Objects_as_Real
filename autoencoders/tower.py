import math
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, Variable
from chainer.initializers import Normal

class text_encoder(chainer.Chain):
    def __init__(self, latent_size=64, num_objects=10, num_descriptions=10):
        super(text_encoder, self).__init__(
            l1=L.Linear(num_objects + num_descriptions, 4 * latent_size, initialW=Normal(0.02)),
            norm1 = L.BatchNormalization(4 * latent_size),
            l2=L.Linear(4 * latent_size, 4 * latent_size, initialW=Normal(0.02)),
            norm2 = L.BatchNormalization(4 * latent_size),
            mean=L.Linear(4 * latent_size, latent_size, initialW=Normal(0.02)),
            var=L.Linear(4 * latent_size, latent_size, initialW=Normal(0.02)),
        )

    def __call__(self, objects_one_hot, descs_one_hot, train=True):
        with chainer.using_config('train', train), chainer.using_config('enable_backprop', train):
            xp = cuda.get_array_module(objects_one_hot.data)
            h1 = F.leaky_relu(self.norm1(self.l1(F.concat((objects_one_hot, descs_one_hot), axis=-1))))
            h2 = F.leaky_relu(self.norm2(self.l2(h1)))

            mean = self.mean(h2)
            var = F.tanh(self.var(h2))
            rand = xp.random.normal(0, 1, var.data.shape).astype(np.float32)
            z = mean + F.exp(var) * Variable(rand)

            return z, mean, var

class text_generator(chainer.Chain):
    def __init__(self, latent_size=64, num_objects=10, num_descriptions=10):
        super(text_generator, self).__init__(
            l3=L.Linear(latent_size, 4 * latent_size, initialW=Normal(0.02)),
            norm3 = L.BatchNormalization(4 * latent_size),
            l2=L.Linear(4 * latent_size, 4 * latent_size, initialW=Normal(0.02)),
            norm2 = L.BatchNormalization(4 * latent_size),
            l1_0=L.Linear(4 * latent_size, num_objects, initialW=Normal(0.02)),
            l1_1=L.Linear(4 * latent_size, num_descriptions, initialW=Normal(0.02)),
        )

    def __call__(self, latent, train=True):
        with chainer.using_config('train', train), chainer.using_config('enable_backprop', train):
            xp = cuda.get_array_module(latent.data)
            h3 = F.leaky_relu(self.norm3(self.l3(latent)))
            h2 = F.leaky_relu(self.norm2(self.l2(h3)))
            return self.l1_0(h2), self.l1_1(h2)

class Encoder_double_z(chainer.Chain):
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
    def __init__(self, density=1, size=64, latent_size=100, channel=3):
        """
        density - a scaling factor for the number of channels in the convolutional layers. It is multiplied by at least
        16,32,64 and 128 as we go deeper. 
        Use: using density=8 when training the VAE separately. using density=4 when training end to end
        Intent: increase the number of features in the convolutional layers. 
        """
        assert (size % 16 == 0)
        self.second_size = size // 4
        initial_size = size // 16
        super(Encoder_double_z, self).__init__(
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
            dc4=L.Convolution2D(int(64 * density), int(128 * density), 3, stride=2, pad=1,
                                initialW=Normal(0.02)),
            norm4=L.BatchNormalization(int(128 * density)),
            mean_robot=L.Linear(initial_size * initial_size * int(128 * density), latent_size,
                          initialW=Normal(0.02)),
            # var_robot=L.Linear(initial_size * initial_size * int(128 * density), latent_size,
            #              initialW=Normal(0.02)),
            mean_scene=L.Linear(initial_size * initial_size * int(128 * density), latent_size,
                          initialW=Normal(0.02)),
            # var_scene=L.Linear(initial_size * initial_size * int(128 * density), latent_size,
            #              initialW=Normal(0.02)),
        )

    def __call__(self, x, train=True):
        with chainer.using_config('train', train), chainer.using_config('enable_backprop', train):
            xp = cuda.get_array_module(x.data)
            h1 = F.leaky_relu(self.dc1(x))
            h1_ = F.leaky_relu(self.dc1_(h1))
            h2 = F.leaky_relu(self.norm2(self.dc2(h1_)))
            h2_ = F.leaky_relu(self.norm2_(self.dc2_(h2)))
            h2_p = F.leaky_relu(self.norm2_p(self.dc2_p(h2_)))
            h2_ = h2_ + h2_p
            h3 = F.leaky_relu(self.norm3(self.dc3(h2_)))
            h3_ = F.leaky_relu(self.norm3_(self.dc3_(h3)))
            h3_p = F.leaky_relu(self.norm3_p(self.dc3_p(h3)))
            h3_ = h3_ + h3_p
            h4 = F.leaky_relu(self.norm4(self.dc4(h3_)))

            mean_robot = self.mean_robot(h4)
            # var_robot = F.tanh(self.var_robot(h4))
            # rand = xp.random.normal(0, 1, var_robot.data.shape).astype(np.float32)
            # z_robot = mean_robot + F.exp(var_robot) * Variable(rand)

            mean_scene = self.mean_scene(h4)
            # var_scene = F.tanh(self.var_scene(h4))
            # rand = xp.random.normal(0, 1, var_scene.data.shape).astype(np.float32)
            # z_scene = mean_scene + F.exp(var_scene) * Variable(rand)

            return F.normalize(mean_robot, axis=1), F.normalize(mean_scene, axis=1)

class Encoder(chainer.Chain):
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
    def __init__(self, density=1, size=64, latent_size=100, channel=3):
        """
        density - a scaling factor for the number of channels in the convolutional layers. It is multiplied by at least
        16,32,64 and 128 as we go deeper. 
        Use: using density=8 when training the VAE separately. using density=4 when training end to end
        Intent: increase the number of features in the convolutional layers. 
        """
        assert (size % 16 == 0)
        self.second_size = size // 4
        initial_size = size // 16
        super(Encoder, self).__init__(
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
            dc4=L.Convolution2D(int(64 * density), int(128 * density), 3, stride=2, pad=1,
                                initialW=Normal(0.02)),
            norm4=L.BatchNormalization(int(128 * density)),
            mean=L.Linear(initial_size * initial_size * int(128 * density), latent_size,
                          initialW=Normal(0.02)),
            var=L.Linear(initial_size * initial_size * int(128 * density), latent_size,
                         initialW=Normal(0.02)),
        )

    def __call__(self, x, train=True):
        with chainer.using_config('train', train), chainer.using_config('enable_backprop', train):
            xp = cuda.get_array_module(x.data)
            h1 = F.leaky_relu(self.dc1(x))
            h1_ = F.leaky_relu(self.dc1_(h1))
            h2 = F.leaky_relu(self.norm2(self.dc2(h1_)))
            h2_ = F.leaky_relu(self.norm2_(self.dc2_(h2)))
            h2_p = F.leaky_relu(self.norm2_p(self.dc2_p(h2_)))
            h2_ = h2_ + h2_p
            h3 = F.leaky_relu(self.norm3(self.dc3(h2_)))
            h3_ = F.leaky_relu(self.norm3_(self.dc3_(h3)))
            h3_p = F.leaky_relu(self.norm3_p(self.dc3_p(h3)))
            h3_ = h3_ + h3_p
            h4 = F.leaky_relu(self.norm4(self.dc4(h3_)))
            mean = self.mean(h4)
            var = F.tanh(self.var(h4))
            rand = xp.random.normal(0, 1, var.data.shape).astype(np.float32)
            z = mean + F.exp(var) * Variable(rand)
            return z, mean, var, h3_p

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
    def __init__(self, density=1, size=64, latent_size=100, channel=3, hidden_dim=100, num_objects=10, num_descriptions=10):
        """
        density - a scaling factor for the number of channels in the convolutional layers. It is multiplied by at least
        16,32,64 and 128 as we go deeper. 
        Use: using density=8 when training the VAE separately. using density=4 when training end to end
        Intent: increase the number of features in the convolutional layers. 
        """
        assert (size % 16 == 0)
        self.second_size = size // 4
        initial_size = size // 16
        super(Encoder_text_tower, self).__init__(
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
            dc3=L.Convolution2D(int(32 * density + 7), int(64 * density), 3, stride=2, pad=1,
                                initialW=Normal(0.02)),
            norm3=L.BatchNormalization(int(64 * density)),
            dc3_=L.Convolution2D(int(64 * density), int(64 * density), 3, stride=1, pad=1,
                                initialW=Normal(0.02)),
            norm3_=L.BatchNormalization(int(64 * density)),
            dc3_p=L.Convolution2D(int(64 * density), int(64 * density), 3, stride=1, pad=1,
                                initialW=Normal(0.02)),
            norm3_p=L.BatchNormalization(int(64 * density)),
            dc4=L.Convolution2D(int(64 * density), int(128 * density), 3, stride=2, pad=1,
                                initialW=Normal(0.02)),
            norm4=L.BatchNormalization(int(128 * density)),
            toConv=L.Linear(num_objects + num_descriptions, self.second_size * self.second_size * 7, initialW=Normal(0.02)),
            norm_toConv=L.BatchNormalization(7),
            mean=L.Linear(initial_size * initial_size * int(128 * density), latent_size,
                          initialW=Normal(0.02)),
            var=L.Linear(initial_size * initial_size * int(128 * density), latent_size,
                         initialW=Normal(0.02)),
        )

    def __call__(self, x, objects_one_hot, descs_one_hot, train=True):
        with chainer.using_config('train', train), chainer.using_config('enable_backprop', train):
            xp = cuda.get_array_module(x.data)
            # h0 = F.concat((x, objects, descs), axis=1)
            h0 = self.toConv(F.concat((objects_one_hot, descs_one_hot), axis=-1))
            h0 = F.reshape(h0, (h0.shape[0], 7, self.second_size, self.second_size))
            h0 = F.leaky_relu(self.norm_toConv(h0))
            h1 = F.leaky_relu(self.dc1(x))
            h1_ = F.leaky_relu(self.dc1_(h1))
            h2 = F.leaky_relu(self.norm2(self.dc2(h1_)))
            h2_ = F.leaky_relu(self.norm2_(self.dc2_(h2)))
            h2_p = F.leaky_relu(self.norm2_p(self.dc2_p(h2_)))
            h2_ = h2_ + h2_p
            h2_ = F.concat((h2_, h0), axis=1)
            h3 = F.leaky_relu(self.norm3(self.dc3(h2_)))
            h3_ = F.leaky_relu(self.norm3_(self.dc3_(h3)))
            h3_p = F.leaky_relu(self.norm3_p(self.dc3_p(h3)))
            h3_ = h3_ + h3_p
            h4 = F.leaky_relu(self.norm4(self.dc4(h3_)))
            mean = self.mean(h4)
            var = F.tanh(self.var(h4))
            rand = xp.random.normal(0, 1, var.data.shape).astype(np.float32)
            z = mean + F.exp(var) * Variable(rand)
            return z, mean, var, h4


class Generator_text(chainer.Chain):
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
    def __init__(self, density=1, size=64, latent_size=100, channel=3, num_objects=10, num_descriptions=10):
        filter_size = 2
        self.intermediate_size = size // 8
        assert (size % 16 == 0)
        initial_size = size // 16
        super(Generator_text, self).__init__(
            g0=L.Linear(num_objects + num_descriptions, self.intermediate_size * self.intermediate_size * 7, initialW=Normal(0.02)),
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
            g3=L.Deconvolution2D(int(64 * density + 7), int(32 * density), filter_size, stride=2, pad=0,
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

    def __call__(self, z, objs, descs, train=True):
        with chainer.using_config('train', train), chainer.using_config('enable_backprop', train):
            h0 = self.g0(F.concat((objs, descs), axis=-1))
            h0 = F.reshape(h0, (h0.shape[0], 7, self.intermediate_size, self.intermediate_size))
            h1 = F.reshape(F.relu(self.norm1(self.g1(z))),
                           (z.data.shape[0], int(128 * self.density), self.initial_size, self.initial_size))
            h2 = F.relu(self.norm2(self.g2(h1)))
            h2_ = F.relu(self.norm2_(self.g2_(h2)))
            h2_p = F.relu(self.norm2_p(self.g2_p(h2_)))
            h2_ = h2_ + h2_p
            h2_ = F.concat((h2_, h0), axis=1)
            h3 = F.relu(self.norm3(self.g3(h2_)))
            h3_ = F.relu(self.norm3_(self.g3_(h3)))
            h3_p = F.relu(self.norm3_p(self.g3_p(h3_)))
            h3_ = h3_ + h3_p
            h4 = F.relu(self.norm4(self.g4(h3_)))
            h4_ = F.relu(self.norm4_(self.g4_(h4)))
            return F.tanh(self.g5(h4_))

class Generator_text_att(chainer.Chain):
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
    def __init__(self, density=1, size=64, latent_size=100, channel=3, num_objects=10, num_descriptions=10):
        filter_size = 2
        self.size = size
        self.intermediate_size = size // 8
        assert (size % 16 == 0)
        self.att_size = 5
        initial_size = size // 16
        super(Generator_text_att, self).__init__(
            FC0=L.Linear(num_objects + num_descriptions, 128, initialW=Normal(0.02), nobias=True),
            FC1=L.Linear(128, 16, initialW=Normal(0.02), nobias=True),
            FC2=L.Linear(16, 32, initialW=Normal(0.02), nobias=True),
            FC00=L.Linear(64 * 8 * 8, 32 * self.att_size * self.att_size, initialW=Normal(0.02), nobias=True),
            FC01=L.Linear(32 * self.att_size * self.att_size, 16 * self.att_size * self.att_size, initialW=Normal(0.02), nobias=True),
            FC02=L.Linear(16 * self.att_size * self.att_size, 32 * self.att_size * self.att_size, initialW=Normal(0.02), nobias=True),
            FC22=L.Linear(32 * self.att_size * self.att_size, 1 * self.att_size * self.att_size, initialW=Normal(0.02), nobias=True),
            att_norm0=L.BatchNormalization(128),
            att_norm1=L.BatchNormalization(16),
            att_norm2=L.BatchNormalization(32),
            att_norm00=L.BatchNormalization(32),
            att_norm01=L.BatchNormalization(16),
            att_norm02=L.BatchNormalization(32),
            att_norm11=L.BatchNormalization(32),
            g2_extra=L.Convolution2D(int(512), int(64), 3, stride=2, pad=1,
                                initialW=Normal(0.02)),
            norm2_extra=L.BatchNormalization(int(64)),
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
            norm2_att=L.BatchNormalization(int(16 * density)),
            g1_att=L.Linear(self.att_size * self.att_size * int(16 * density), self.att_size * self.att_size * 1, initialW=Normal(0.02)),
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

    def __call__(self, z, objs, descs, features, train=True):
        with chainer.using_config('train', train), chainer.using_config('enable_backprop', train):
            h1 = F.relu(self.norm1(self.g1(z)))
            h1 = F.reshape(h1, (z.data.shape[0], int(128 * self.density), self.initial_size, self.initial_size))
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
            
            #Attention Part
            att_h0 = F.tanh(self.att_norm0(self.FC0(F.concat((objs, descs), axis=-1))))
            att_h1 = F.tanh(self.att_norm1(self.FC1(att_h0)))
            att_h2 = F.tanh(self.att_norm2(self.FC2(att_h1)))
            att_h2 = F.tile(F.expand_dims(att_h2, axis=2), (1, 1, self.att_size * self.att_size))

            # features = F.transpose(features, (0, 2, 3, 1))
            # features = F.reshape(features, (-1, features.shape[1] * features.shape[2], features.shape[3]))

            features = F.leaky_relu(self.norm2_extra(self.g2_extra(features)))
            att_f0 = F.reshape(self.FC00(features), (-1, 32, self.att_size * self.att_size))
            att_f0 = F.tanh(self.att_norm00(att_f0))
            original_features = att_f0
            
            att_f1 = F.reshape(self.FC01(att_f0), (-1, 16, self.att_size * self.att_size))
            att_f1 = F.tanh(self.att_norm01(att_f1))

            att_f2 = F.reshape(self.FC02(att_f1), (-1, 32, self.att_size * self.att_size))
            att_f2 = F.tanh(self.att_norm02(att_f2))

            att_ff = att_h2 + att_f2
            att_ff = F.tanh(self.att_norm11(att_ff))
            
            att_att = F.reshape(self.FC22(att_ff), (-1, self.att_size * self.att_size))
            att_att = F.softmax(att_att, axis=1)
            h1_att = F.reshape(att_att, (-1, 1, self.att_size, self.att_size))

            pooled_features = F.einsum('ijk,ik -> ij', original_features, att_att)
            # h0 = F.expand_dims(h0, axis=2)
            # h0 = F.reshape(F.tile(h0, (1, 1, self.att_size * self.att_size)), (-1, h0.shape[1], self.att_size, self.att_size))
            # h2_att = F.tanh(self.norm2_extra(self.g2_extra(features)))
            # h2_att_orig = F.reshape(features, (-1, int(512), self.att_size * self.att_size))
            # h2_comb = F.tanh(self.norm2_att(h2_att + h0))
            # h2_comb = F.reshape(h2_comb, (-1, int(16 * self.density), self.att_size * self.att_size))
            # h1_b = self.g1_att(h2_comb)
            # h1_att = F.softmax(h1_b/16, axis=1)
            # h1_att = F.reshape(h1_att, (-1, 1, self.att_size, self.att_size))
            # # h1_att_fit = F.unpooling_2d(h1_att, 4, outsize=(16,16))
            # h1_att_fit = F.reshape(h1_att, (-1, self.att_size * self.att_size))
            # pooled_features = F.einsum('ijk,ik -> ij', h2_att_orig, h1_att_fit)
            
            return F.tanh(self.g5(h4_)), F.resize_images(h1_att, (self.size, self.size)), h1_att, pooled_features

class Generator_latent_att(chainer.Chain):
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
    def __init__(self, density=1, size=64, latent_size=100, channel=3, num_objects=10, num_descriptions=10):
        filter_size = 2
        self.size = size
        self.intermediate_size = size // 8
        assert (size % 16 == 0)
        self.att_size = 16
        initial_size = size // 16
        super(Generator_latent_att, self).__init__(
            g0_att=L.Linear(latent_size, 512, initialW=Normal(0.02)),
            g0=L.Linear(num_objects + num_descriptions, 64, initialW=Normal(0.02)),
            norm0=L.BatchNormalization(64),
            g00=L.Linear(latent_size, 64, initialW=Normal(0.02)),
            norm00=L.BatchNormalization(64),
            norm00_att=L.BatchNormalization(512),
            norm000=L.BatchNormalization(64),
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
            g2_extra=L.Deconvolution2D(int(64 * density), int(64 * density), 3, stride=1, pad=1,
                                initialW=Normal(0.02)),
            norm2_extra=L.BatchNormalization(int(64 * density)),
            norm2_att=L.BatchNormalization(int(64 * density)),
            g1_att=L.Linear(16 * 16 * int(64 * density), self.att_size * self.att_size * 1, initialW=Normal(0.02)),
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

    def __call__(self, z, objs, descs, features, train=True):
        with chainer.using_config('train', train), chainer.using_config('enable_backprop', train):
            h0 = F.tanh(self.norm0(self.g0(F.concat((objs, descs), axis=-1))))
            h00 = F.tanh(self.norm00(self.g00(z)))
            z_mask = F.softmax(self.norm000(h00 + h0))
            z_pr = z * z_mask
            
            h1 = F.relu(self.norm1(self.g1(z_pr)))
            h1 = F.reshape(h1, (z.data.shape[0], int(128 * self.density), self.initial_size, self.initial_size))
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

            #Attention Part
            h000 = F.tanh(self.norm00_att(self.g0_att(z_pr)))
            h0000 = F.expand_dims(h000, axis=2)
            h0000 = F.reshape(F.tile(h0000, (1, 1, 16 * 16)), (-1, h0000.shape[1], 16, 16))
            h2_att = F.tanh(self.norm2_extra(self.g2_extra(h2_)))
            h2_att_orig = F.reshape(h2_, (-1, int(64 * self.density), 16 * 16))
            h2_comb = F.tanh(self.norm2_att(h2_att + h0000))
            h2_comb = F.reshape(h2_comb, (-1, int(64 * self.density), 16 * 16))
            h1_b = self.g1_att(h2_comb)
            h1_att = F.softmax(h1_b, axis=1)
            h1_att = F.reshape(h1_att, (-1, 1, self.att_size, self.att_size))
            # h1_att_fit = F.unpooling_2d(h1_att, 4, outsize=(16,16))
            h1_att_fit = F.reshape(h1_att, (-1, 16 * 16))
            pooled_features = F.einsum('ijk,ik -> ij', h2_att_orig, h1_att_fit)
            
            return F.tanh(self.g5(h4_)), F.resize_images(h1_att, (self.size, self.size)), h1_att, pooled_features

class Discriminator_texual(chainer.Chain):
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
    def __init__(self, density=1, size=64, channel=3, num_words=32, num_objects=10, num_descriptions=10):
        assert (size % 16 == 0)
        self.num_objects = num_objects
        self.num_descriptions = num_descriptions
        initial_size = size // 16
        super(Discriminator_texual, self).__init__(
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
            dc5=L.Linear(initial_size * initial_size * int(128 * density), num_objects, initialW=Normal(0.02)),
            dc6=L.Linear(initial_size * initial_size * int(128 * density), num_descriptions, initialW=Normal(0.02)),
            dc8=L.Linear(initial_size * initial_size * int(128 * density), num_objects, initialW=Normal(0.02)),
            dc9=L.Linear(initial_size * initial_size * int(128 * density), num_descriptions, initialW=Normal(0.02)),
        )

    def __call__(self, x, att=True, train=True):
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
            if att:
                return self.dc5(h4), self.dc6(h4), h3
            else:
                return self.dc8(h4), self.dc9(h4), h3

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
            dc6=L.Linear(initial_size * initial_size * int(128 * density), 2, initialW=Normal(0.02)),
        )

    def __call__(self, x, att=True, train=True):
        with chainer.using_config('train', train):
            h1 = F.leaky_relu(self.dc1(x))
            h2 = F.leaky_relu(self.norm2(self.dc2(h1)))
            h2_ = F.leaky_relu(self.norm2_(self.dc2_(h2)))
            h2_p = F.leaky_relu(self.norm2_p(self.dc2_p(h2_)))
            h2_p = h2_ + h2_p
            h3 = F.leaky_relu(self.norm3(self.dc3(h2_p)))
            h3_ = F.leaky_relu(self.norm3_(self.dc3_(h3)))
            h3_p = F.leaky_relu(self.norm3_p(self.dc3_p(h3_)))
            h3_p = h3_ + h3_p
            h4 = F.leaky_relu(self.norm4(self.dc4(h3_p)))
            if att:
                return self.dc5(h4), h3
            else:
                return self.dc6(h4), h3
