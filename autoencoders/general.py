import math
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, Variable
from chainer.initializers import Normal

class image2image_fc(chainer.Chain):
    def __init__(self, size=64, channel=3):
        super(image2image_fc, self).__init__(
            convertor=L.Linear(size * size * channel, size * size * channel, initialW=Normal(0.02)),
        )

    def __call__(self, x, train=True):
        with chainer.using_config('train', train), chainer.using_config('enable_backprop', train):
            out = self.convertor(F.reshape(x, (x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])))
            return F.reshape(out, x.shape)

class image2image_conv(chainer.Chain):
    def __init__(self, density=1, size=64, channel=3):
        super(image2image_conv, self).__init__(
            dc1=L.Convolution2D(channel, int(16 * density), 3, stride=2, pad=1,
                                initialW=Normal(0.02)),
            dc2=L.Convolution2D(int(16 * density), int(32 * density), 3, stride=2, pad=1,
                                initialW=Normal(0.02)),
            dc3=L.Convolution2D(int(32 * density), int(32 * density), 3, stride=1, pad=1,
                                initialW=Normal(0.02)),
            norm3=L.BatchNormalization(int(32 * density)),
            g4=L.Deconvolution2D(int(32 * density), int(16 * density), 2, stride=2, pad=0,
                                 initialW=Normal(0.02)),
            norm4=L.BatchNormalization(int(16 * density)),
            g5=L.Deconvolution2D(int(16 * density), channel, 2, stride=2, pad=0,
                                 initialW=Normal(0.02)),
        )

    def __call__(self, x, train=True):
        with chainer.using_config('train', train), chainer.using_config('enable_backprop', train):
            h1 = F.leaky_relu(self.dc1(x))
            h2 = F.leaky_relu(self.dc2(h1))
            h3 = F.leaky_relu(self.norm3(self.dc3(h2)))
            h4 = F.relu(self.norm4(self.g4(h3)))
            return F.tanh(self.g5(h4))


class classifier_simple(chainer.Chain):
    def __init__(self, size=128, num_objects=10, num_descriptions=10):
        super(classifier_simple, self).__init__(
            dc0=L.Linear(size, 128, initialW=Normal(0.02)),
            dc1=L.Linear(128, 64, initialW=Normal(0.02)),
            dc2=L.Linear(64, 64, initialW=Normal(0.02)),
            dc3=L.Linear(64, 32, initialW=Normal(0.02)),
            dc4=L.Linear(32, 32, initialW=Normal(0.02)),

            dc5=L.Linear(32, num_objects, initialW=Normal(0.02)),
            dc55=L.Linear(32, num_descriptions, initialW=Normal(0.02)),
        )

    def __call__(self, x, train=True):
        with chainer.using_config('train', train):
            h0 = self.dc0(x)
            h1 = self.dc1(h0)
            h2 = self.dc2(h1)
            h3 = self.dc3(h2)
            h4 = self.dc4(h3)

            h00 = self.dc5(h4)
            h11 = self.dc55(h4)

            return h00, h11

class Discriminator_simple(chainer.Chain):
    def __init__(self, density=1, size=5, channel=1):
        super(Discriminator_simple, self).__init__(
            dc1=L.Convolution2D(channel, int(16 * density), 2, stride=1, pad=1,
                                initialW=Normal(0.02)),
            dc2=L.Convolution2D(int(16 * density), int(16 * density), 2, stride=1, pad=1,
                                initialW=Normal(0.02)),
            norm2=L.BatchNormalization(int(16 * density)),
            dc3=L.Convolution2D(int(16 * density), int(32 * density), 2, stride=2, pad=1,
                                initialW=Normal(0.02)),
            norm3=L.BatchNormalization(int(32 * density)),
            dc4=L.Convolution2D(int(32 * density), int(32 * density), 2, stride=1, pad=0,
                                initialW=Normal(0.02)),
            norm4=L.BatchNormalization(int(32 * density)),
            dc5=L.Linear(3 * 3 * int(32 * density), 2, initialW=Normal(0.02)),
        )

    def __call__(self, x, train=True):
        with chainer.using_config('train', train):
            h1 = F.leaky_relu(self.dc1(x))
            h2 = F.leaky_relu(self.norm2(self.dc2(h1)))
            h3 = F.leaky_relu(self.norm3(self.dc3(h2)))
            h4 = F.leaky_relu(self.norm4(self.dc4(h3)))

            return self.dc5(h4), h4