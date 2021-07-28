import random

import numpy as np
import torch
from scipy.ndimage import rotate, map_coordinates, gaussian_filter
from scipy.ndimage.filters import convolve
from torchvision.transforms import Compose

MAX = max


class RandomFlip:
    """
    Randomly flips the image across the given axes. Image can be either 3D (DxHxW) or 4D (CxDxHxW).

    When creating make sure that the provided RandomStates are consistent between raw and labeled datasets,
    otherwise the models won't converge.
    """

    def __init__(self, random_state, axes=(0, 2)):
        assert random_state is not None, 'RandomState cannot be None'
        self.random_state = random_state
        # self.axes = (0, 1, 2)
        self.axes = axes

    def __call__(self, m):
        assert m.ndim in [3, 4], 'Supports only 3D (DxHxW) or 4D (CxDxHxW) images'

        for axis in self.axes:
            if self.random_state.uniform() > 0.5:
                if m.ndim == 3:
                    m = np.flip(m, axis)
                else:
                    channels = [np.flip(m[c], axis) for c in range(m.shape[0])]
                    m = np.stack(channels, axis=0)

        return m


class RandomRotate90:
    """
    Rotate an array by 90 degrees around a randomly chosen plane. Image can be either 3D (DxHxW) or 4D (CxDxHxW).

    When creating make sure that the provided RandomStates are consistent between raw and labeled datasets,
    otherwise the models won't converge.

    IMPORTANT: assumes DHW axis order (that's why rotation is performed across (1,2) axis)
    """

    def __init__(self, random_state, axes=(1, 2)):
        self.random_state = random_state
        self.axes = axes

    def __call__(self, m):
        assert m.ndim in [3, 4], 'Supports only 3D (DxHxW) or 4D (CxDxHxW) images'

        # pick number of rotations at random
        k = self.random_state.randint(0, 4)
        # rotate k times around a given plane
        if m.ndim == 3:
            m = np.rot90(m, k, self.axes)
        else:
            channels = [np.rot90(m[c], k, self.axes) for c in range(m.shape[0])]
            m = np.stack(channels, axis=0)

        return m


class RandomRotate:
    """
    Rotate an array by a random degrees from taken from (-angle_spectrum, angle_spectrum) interval.
    Rotation axis is picked at random from the list of provided axes.
    """

    def __init__(self, random_state, angle_spectrum=10, axes=None, mode='constant', cval=-1):
        if axes is None:
            axes = [(1, 0), (2, 1), (2, 0)]
        else:
            assert isinstance(axes, list) and len(axes) > 0

        self.random_state = random_state
        self.angle_spectrum = angle_spectrum
        self.axes = axes
        self.mode = mode
        self.cval = cval

    def __call__(self, m):
        axis = self.axes[self.random_state.randint(len(self.axes))]
        angle = self.random_state.randint(-self.angle_spectrum, self.angle_spectrum)

        if m.ndim == 3:
            m = rotate(m, angle, axes=axis, reshape=False, order=0, mode=self.mode, cval=self.cval)
        else:
            channels = [rotate(m[c], angle, axes=axis, reshape=False, order=0, mode=self.mode, cval=self.cval) for c in
                        range(m.shape[0])]
            m = np.stack(channels, axis=0)

        return m


class RandomContrast:
    """
       Adjust the contrast of an image by a random factor inside a the contrast_range
    """

    def __init__(self, random_state, contrast_range=(0.25, 1.0), execution_probability=0.2):
        assert len(contrast_range) == 2
        self.min_factor, self.max_factor = contrast_range
        self.random_state = random_state
        self.execution_probability = execution_probability

    def __call__(self, m):
        factor = self.random_state.uniform(self.min_factor, self.max_factor)

        if self.random_state.uniform() < self.execution_probability:
            if m.ndim == 3:
                # take the mean intensity of the entire patch
                mean_intensity = np.mean(m)
            else:
                # if 4D then compute per channel mean intensity (assuming: CZYX axis order)
                mean_intensity = np.mean(m, axis=(1, 2, 3))
            return np.clip(mean_intensity + factor * (m - mean_intensity), -1, 1)

        return m


class RandomBrightness:
    """
        Adjust the brightness of an image by a random factor inside a the brightness_range
        Brightness range: tuple,float. If it's a tuple a random factor will be taken from (brightness_range[0], brightness_range[1])
        If it's float then the random factor will be taken from (-brightness_range,brightness_range).
        The intervals must be included in [-1,1]. If not, they would be clipped to [-1,1]
    """

    def __init__(self, random_state, brightness_range=0.1):
        if isinstance(brightness_range, tuple):
            assert len(brightness_range) == 2
            self.brightness_min, self.brightness_max = np.clip(brightness_range, -1., 1.)
        else:
            self.brightness_min, self.brightness_max = np.clip([-brightness_range, brightness_range], -1., 1.)
        self.random_state = random_state

    def __call__(self, m):
        brightness = self.random_state.uniform(self.brightness_min, self.brightness_max)
        return np.clip(m + brightness, -1, 1)


class RandomBrightnessContrast:
    """
        Apply RandomBrightness and RandomContrast interchangeably
    """

    def __init__(self, random_state, brightness_range=0.1, contrast_range=(0.25, 1.0)):
        self.rand_contrast = RandomContrast(random_state, contrast_range)
        self.rand_brightness = RandomBrightness(random_state, brightness_range)
        self.random_state = random_state

    def __call__(self, m):
        if self.random_state.uniform() < 0.5:  # Alternates order of the Brightness and Contrast transforms
            m = self.rand_brightness(m)
            return self.rand_contrast(m)
        else:
            m = self.rand_contrast(m)
            return self.rand_brightness(m)


# it's relatively slow, i.e. ~1s per patch of size 64x200x200
class ElasticDeformation:
    """
    Apply elasitc deformations of 3D patches on a per-voxel mesh. Assumes ZYX axis order!
    Based on: https://github.com/fcalvet/image_tools/blob/master/image_augmentation.py#L62
    """

    def __init__(self, random_state, spline_order, alpha=15, sigma=3):
        """
        :param spline_order: the order of spline interpolation (use 0 for labeled images)
        :param alpha: scaling factor for deformations
        :param sigma: smothing factor for Gaussian filter
        """
        self.random_state = random_state
        self.spline_order = spline_order
        self.alpha = alpha
        self.sigma = sigma

    def __call__(self, m):
        assert m.ndim == 3

        dz = gaussian_filter(self.random_state.randn(*m.shape), self.sigma, mode="constant", cval=0) * self.alpha
        dy = gaussian_filter(self.random_state.randn(*m.shape), self.sigma, mode="constant", cval=0) * self.alpha
        dx = gaussian_filter(self.random_state.randn(*m.shape), self.sigma, mode="constant", cval=0) * self.alpha

        z_dim, y_dim, x_dim = m.shape
        z, y, x = np.meshgrid(np.arange(z_dim), np.arange(y_dim), np.arange(x_dim), indexing='ij')
        indices = z + dz, y + dy, x + dx
        return map_coordinates(m, indices, order=self.spline_order, mode='reflect')


class AbstractLabelToBoundary:
    AXES_TRANSPOSE = [
        (0, 1, 2),  # X
        (0, 2, 1),  # Y
        (2, 0, 1)  # Z
    ]

    def __init__(self, ignore_index=None):
        """
        :param ignore_index: label to be ignored in the output, i.e. after computing the boundary the label ignore_index
            will be restored where is was in the patch originally
        """
        self.ignore_index = ignore_index

    def __call__(self, m):
        """
        Extract boundaries from a given 3D label tensor.
        :param m: input 3D tensor
        :return: binary mask, with 1-label corresponding to the boundary and 0-label corresponding to the background
        """
        assert m.ndim == 3

        kernels = self.get_kernels()
        assert len(kernels) % 3 == 0, "Number of kernels must be divided by 3 (one kernel per offset per Z,Y,X axes"
        channels = np.stack([np.where(np.abs(convolve(m, kernel)) > 0, 1, 0) for kernel in kernels])
        results = []
        # aggregate affinities with the same offset
        for i in range(0, len(kernels), 3):
            xyz_aggregated_affinities = np.logical_or.reduce(channels[i:i + 3, ...]).astype(np.int)
            xyz_aggregated_affinities = self._recover_ignore_index(xyz_aggregated_affinities, m)
            results.append(xyz_aggregated_affinities)

        if len(results) > 1:
            # stack if more than one channel
            return np.stack(results, axis=0)
        else:
            # otherwise just take first channel
            return results[0]

    @staticmethod
    def create_kernel(axis, offset):
        # create conv kernel
        k_size = offset + 1
        k = np.zeros((1, 1, k_size), dtype=np.int)
        k[0, 0, 0] = 1
        k[0, 0, offset] = -1
        return np.transpose(k, axis)

    def _recover_ignore_index(self, input, orig):
        if self.ignore_index is not None:
            mask = orig == self.ignore_index
            input[mask] = self.ignore_index

        return input

    def get_kernels(self):
        raise NotImplementedError


class RandomLabelToBoundary(AbstractLabelToBoundary):
    """
    Converts a given volumetric label array to binary mask corresponding to borders between labels.
    One specify the max_offset (thickness) of the border. Then the offset is picked at random every time you call
    the transformer (offset is picked form the range 1:max_offset) for each axis and the boundary computed.
    One may use this scheme in order to make the network more robust against various thickness of borders in the ground
    truth  (think of it as a boundary denoising scheme).
    """

    def __init__(self, random_state, max_offset=8, ignore_index=None):
        super().__init__(ignore_index)
        self.random_state = random_state
        self.offsets = tuple(range(1, max_offset + 1))

    def get_kernels(self):
        return [self.create_kernel(axis, random.choice(self.offsets)) for axis in self.AXES_TRANSPOSE]


class LabelToBoundary(AbstractLabelToBoundary):
    """
    Converts a given volumetric label array to binary mask corresponding to borders between labels.
    One specify the offsets (thickness) of the border. The boundary will be computed via the convolution operator.
    """

    def __init__(self, offsets, ignore_index=None):
        super().__init__(ignore_index)
        if isinstance(offsets, int):
            assert offsets > 0, "'offset' must be positive"
            offsets = [offsets]
        elif isinstance(offsets, list) or isinstance(offsets, tuple):
            assert all(a > 0 for a in offsets), "'offset' must be positive"
            assert len(set(offsets)) == len(offsets), "'offsets' must be unique"
        else:
            raise ValueError(f"Unsupported 'offsets' type {type(offsets)}")

        self.kernels = []
        # create kernel for every axis-offset pair
        for offset in offsets:
            for axis in self.AXES_TRANSPOSE:
                # create kernels for a given offset in every direction
                self.kernels.append(self.create_kernel(axis, offset))

    def get_kernels(self):
        return self.kernels


class Clip:
    """
    Clip the image range to a certain range
    """

    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, m):
        return np.clip(m, self.min_val, self.max_val)


class Normalize:
    """
    Normalizes a given input tensor to be 0-mean and 1-std.
    mean and std parameter have to be provided explicitly.
    """

    def __init__(self, mean, std, eps=1e-4):
        self.mean = mean
        self.std = std
        self.eps = eps

    def __call__(self, m):
        return (m - self.mean) / (self.std + self.eps)


class NormalizeV2:

    def __call__(self, m):
        return (m - np.min(m)) / (np.max(m) - np.min(m))


class ToTensor:
    """
    Converts a given input numpy.ndarray into torch.Tensor. Adds additional 'channel' axis when the input is 3D
    and expand_dims=True (use for raw data of the shape (D, H, W)).
    """

    def __init__(self, expand_dims, dtype=np.float32):
        self.expand_dims = expand_dims
        self.dtype = dtype

    def __call__(self, m):
        assert m.ndim in [3, 4], 'Supports only 3D (DxHxW) or 4D (CxDxHxW) images'
        # add channel dimension
        if self.expand_dims and m.ndim == 3:
            m = np.expand_dims(m, axis=0)

        return torch.from_numpy(m.astype(dtype=self.dtype))


class Identity:
    def __call__(self, m):
        return m


# Helper Transformer classes

class TransformerBuilder:
    def __init__(self, transformer_class, config):
        self.transformer_class = transformer_class
        self.config = config
        self.mean = None
        self.std = None
        self.phase = None

    @property
    def mean(self):
        return self._mean

    @mean.setter
    def mean(self, mean):
        self._mean = mean

    @property
    def std(self):
        return self._std

    @std.setter
    def std(self, std):
        self._std = std

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, phase):
        self._phase = phase

    def build(self):
        assert self.mean is not None
        assert self.std is not None
        assert self.phase is not None
        assert 'label_dtype' in self.config
        return self.transformer_class.create(self.mean, self.std, self.phase, **self.config)


class BaseTransformer:
    """
    Base transformer class used for data augmentation.
    """
    seed = 47

    def __init__(self, mean, std, phase, label_dtype, **kwargs):
        self.mean = mean
        self.std = std
        self.phase = phase
        self.label_dtype = label_dtype

    def raw_transform(self):
        return Compose([
            # Normalize(self.mean, self.std),
            NormalizeV2(),
            ToTensor(expand_dims=True)
        ])

    def label_transform(self):
        return Compose([
            ToTensor(expand_dims=False, dtype=self.label_dtype)
        ])

    def weight_transform(self):
        return Compose([
            ToTensor(expand_dims=False)
        ])

    @classmethod
    def create(cls, mean, std, phase, label_dtype, **kwargs):
        return cls(mean, std, phase, label_dtype, **kwargs)


class StandardTransformer(BaseTransformer):
    """
    Standard data augmentation: random flips across randomly picked axis + random 90 degrees rotations.

    """

    def __init__(self, mean, std, phase, label_dtype, **kwargs):
        super().__init__(mean=mean, std=std, phase=phase, label_dtype=label_dtype)
        assert 'clip_min' in kwargs, "'clip_min' argument required"
        assert 'clip_max' in kwargs, "'clip_max' argument required"
        self.clip_min = kwargs['clip_min']
        self.clip_max = kwargs['clip_max']

    def raw_transform(self):
        if self.phase == 'train':
            return Compose([
                Clip(self.clip_min, self.clip_max),
                # Normalize(self.mean, self.std),
                NormalizeV2(),
                RandomContrast(np.random.RandomState(self.seed)),
                RandomFlip(np.random.RandomState(self.seed)),
                RandomRotate90(np.random.RandomState(self.seed)),
                ToTensor(expand_dims=True)
            ])
        else:
            return super().raw_transform()

    def label_transform(self):
        if self.phase == 'train':
            return Compose([
                RandomFlip(np.random.RandomState(self.seed)),
                RandomRotate90(np.random.RandomState(self.seed)),
                ToTensor(expand_dims=False, dtype=self.label_dtype)
            ])
        else:
            return super().label_transform()

    def weight_transform(self):
        if self.phase == 'train':
            return Compose([
                RandomFlip(np.random.RandomState(self.seed)),
                RandomRotate90(np.random.RandomState(self.seed)),
                ToTensor(expand_dims=False)
            ])
        else:
            return super().weight_transform()


class MedicalTransformer(BaseTransformer):
    """
    Medical data augmentation: random flips across randomly picked axis + random 90 degrees rotations.

    """

    def __init__(self, mean, std, phase, label_dtype, **kwargs):
        super().__init__(mean=mean, std=std, phase=phase, label_dtype=label_dtype)
        assert 'clip_min' in kwargs, "'clip_min' argument required"
        assert 'clip_max' in kwargs, "'clip_max' argument required"
        self.clip_min = kwargs['clip_min']
        self.clip_max = kwargs['clip_max']

    def raw_transform(self):
        if self.phase == 'train':
            return Compose([
                Clip(self.clip_min, self.clip_max),
                ElasticDeformation(np.random.RandomState(self.seed), 0),
                Normalize(self.mean, self.std),
                RandomContrast(np.random.RandomState(self.seed)),
                ToTensor(expand_dims=True)
            ])
        else:
            return Compose([
                Clip(self.clip_min, self.clip_max),
                Normalize(self.mean, self.std),
                ToTensor(expand_dims=True)
            ])

    def label_transform(self):
        if self.phase == 'train':
            return Compose([
                ElasticDeformation(np.random.RandomState(self.seed), 0),
                ToTensor(expand_dims=False, dtype=self.label_dtype)
            ])
        else:
            return super().label_transform()

    def weight_transform(self):
        if self.phase == 'train':
            return Compose([
                ElasticDeformation(np.random.RandomState(self.seed), 0),
                ToTensor(expand_dims=False)
            ])
        else:
            return super().weight_transform()


class IsotropicRotationTransformer(BaseTransformer):
    """
    Data augmentation to be used with isotropic 3D volumes: random flips across randomly picked axis + random 90 deg
    rotations + random angle rotations across randomly picked axis.
    """

    def __init__(self, mean, std, phase, label_dtype, **kwargs):
        super().__init__(mean=mean, std=std, phase=phase, label_dtype=label_dtype)
        assert 'angle_spectrum' in kwargs, "'angle_spectrum' argument required"
        self.angle_spectrum = kwargs['angle_spectrum']

    def raw_transform(self):
        if self.phase == 'train':
            return Compose([
                Normalize(self.mean, self.std),
                RandomContrast(np.random.RandomState(self.seed)),
                RandomFlip(np.random.RandomState(self.seed)),
                RandomRotate90(np.random.RandomState(self.seed)),
                RandomRotate(np.random.RandomState(self.seed), angle_spectrum=self.angle_spectrum),
                ToTensor(expand_dims=True)
            ])
        else:
            return super().raw_transform()

    def label_transform(self):
        if self.phase == 'train':
            return Compose([
                RandomFlip(np.random.RandomState(self.seed)),
                RandomRotate90(np.random.RandomState(self.seed)),
                RandomRotate(np.random.RandomState(self.seed), angle_spectrum=self.angle_spectrum),
                ToTensor(expand_dims=False, dtype=self.label_dtype)
            ])
        else:
            return super().label_transform()

    def weight_transform(self):
        if self.phase == 'train':
            return Compose([
                RandomFlip(np.random.RandomState(self.seed)),
                RandomRotate90(np.random.RandomState(self.seed)),
                RandomRotate(np.random.RandomState(self.seed), angle_spectrum=self.angle_spectrum),
                ToTensor(expand_dims=False)
            ])
        else:
            return super().weight_transform()


class AnisotropicRotationTransformer(BaseTransformer):
    """
    Data augmentation to be used with anisotropic 3D volumes: random flips across randomly picked axis + random 90 deg
    rotations + random angle rotations across (1,0) axis.
    """

    def __init__(self, mean, std, phase, label_dtype, **kwargs):
        super().__init__(mean=mean, std=std, phase=phase, label_dtype=label_dtype)
        assert 'angle_spectrum' in kwargs, "'angle_spectrum' argument required"
        self.angle_spectrum = kwargs['angle_spectrum']

    def raw_transform(self):
        if self.phase == 'train':
            return Compose([
                Normalize(self.mean, self.std),
                RandomContrast(np.random.RandomState(self.seed)),
                RandomFlip(np.random.RandomState(self.seed)),
                RandomRotate90(np.random.RandomState(self.seed)),
                # rotate in XY only (ZYX axis order is assumed)
                RandomRotate(np.random.RandomState(self.seed), angle_spectrum=self.angle_spectrum, axes=[(2, 1)]),
                ToTensor(expand_dims=True)
            ])
        else:
            return super().raw_transform()

    def label_transform(self):
        if self.phase == 'train':
            return Compose([
                RandomFlip(np.random.RandomState(self.seed)),
                RandomRotate90(np.random.RandomState(self.seed)),
                # rotate in XY only (ZYX axis order is assumed)
                RandomRotate(np.random.RandomState(self.seed), angle_spectrum=self.angle_spectrum, axes=[(2, 1)]),
                ToTensor(expand_dims=False, dtype=self.label_dtype)
            ])
        else:
            return super().label_transform()

    def weight_transform(self):
        if self.phase == 'train':
            return Compose([
                RandomFlip(np.random.RandomState(self.seed)),
                RandomRotate90(np.random.RandomState(self.seed)),
                RandomRotate(np.random.RandomState(self.seed), angle_spectrum=self.angle_spectrum, axes=[(2, 1)]),
                ToTensor(expand_dims=False)
            ])
        else:
            return super().weight_transform()


class AnisotropicElasticRotationTransformer(BaseTransformer):
    """
    Data augmentation to be used with anisotropic 3D volumes: elastic deformations + random flips across randomly picked
    axis + random 90 deg rotations + random angle rotations across (1,0)
    """

    def __init__(self, mean, std, phase, label_dtype, **kwargs):
        super().__init__(mean=mean, std=std, phase=phase, label_dtype=label_dtype)
        assert 'angle_spectrum' in kwargs, "'angle_spectrum' argument required"
        self.angle_spectrum = kwargs['angle_spectrum']

    def raw_transform(self):
        if self.phase == 'train':
            return Compose([
                Normalize(self.mean, self.std),
                ElasticDeformation(np.random.RandomState(self.seed), spline_order=3),
                RandomContrast(np.random.RandomState(self.seed)),
                RandomFlip(np.random.RandomState(self.seed)),
                RandomRotate90(np.random.RandomState(self.seed)),
                # rotate in XY only (ZYX axis order is assumed)
                RandomRotate(np.random.RandomState(self.seed), angle_spectrum=self.angle_spectrum, axes=[(2, 1)]),
                ToTensor(expand_dims=True)
            ])
        else:
            return super().raw_transform()

    def label_transform(self):
        if self.phase == 'train':
            return Compose([
                ElasticDeformation(np.random.RandomState(self.seed), spline_order=0),
                RandomFlip(np.random.RandomState(self.seed)),
                RandomRotate90(np.random.RandomState(self.seed)),
                # rotate in XY only (ZYX axis order is assumed)
                RandomRotate(np.random.RandomState(self.seed), angle_spectrum=self.angle_spectrum, axes=[(2, 1)]),
                ToTensor(expand_dims=False, dtype=self.label_dtype)
            ])
        else:
            return super().label_transform()

    def weight_transform(self):
        if self.phase == 'train':
            return Compose([
                ElasticDeformation(np.random.RandomState(self.seed), spline_order=3),
                RandomFlip(np.random.RandomState(self.seed)),
                RandomRotate90(np.random.RandomState(self.seed)),
                RandomRotate(np.random.RandomState(self.seed), angle_spectrum=self.angle_spectrum, axes=[(2, 1)]),
                ToTensor(expand_dims=False)
            ])
        else:
            return super().weight_transform()


# Make sure to use mode='reflect' when doing RandomRotate otherwise the transform will produce unwanted boundary signal
class LabelToBoundaryTransformer(BaseTransformer):
    def __init__(self, mean, std, phase, label_dtype, **kwargs):
        super().__init__(mean=mean, std=std, phase=phase, label_dtype=label_dtype)
        assert 'angle_spectrum' in kwargs, "'angle_spectrum' argument required"
        self.angle_spectrum = kwargs['angle_spectrum']
        if 'ignore_index' in kwargs:
            self.ignore_index = kwargs['ignore_index']
        else:
            self.ignore_index = None

    def raw_transform(self):
        if self.phase == 'train':
            return Compose([
                Normalize(self.mean, self.std),
                RandomContrast(np.random.RandomState(self.seed)),
                RandomFlip(np.random.RandomState(self.seed)),
                RandomRotate90(np.random.RandomState(self.seed)),
                # rotate in XY only and make sure mode='reflect' is used in order to prevent boundary artifacts
                RandomRotate(np.random.RandomState(self.seed), angle_spectrum=self.angle_spectrum, axes=[(2, 1)],
                             mode='reflect'),
                ToTensor(expand_dims=True)
            ])
        else:
            return super().raw_transform()

    def label_transform(self):
        if self.phase == 'train':
            return Compose([
                RandomFlip(np.random.RandomState(self.seed)),
                RandomRotate90(np.random.RandomState(self.seed)),
                # rotate in XY only and make sure mode='reflect' is used in order to prevent boundary artifacts
                RandomRotate(np.random.RandomState(self.seed), angle_spectrum=self.angle_spectrum, axes=[(2, 1)],
                             mode='reflect'),
                # this will give us 4 output channels with boundary signal
                LabelToBoundary(offsets=(2, 4, 6, 8), ignore_index=self.ignore_index),
                ToTensor(expand_dims=False, dtype=self.label_dtype)
            ])
        else:
            return Compose([
                # this will give us 4 output channels with boundary signal
                LabelToBoundary(offsets=(2, 4, 6, 8), ignore_index=self.ignore_index),
                ToTensor(expand_dims=False, dtype=self.label_dtype)
            ])

    def weight_transform(self):
        if self.phase == 'train':
            return Compose([
                RandomFlip(np.random.RandomState(self.seed)),
                RandomRotate90(np.random.RandomState(self.seed)),
                RandomRotate(np.random.RandomState(self.seed), angle_spectrum=self.angle_spectrum, axes=[(2, 1)],
                             mode='reflect'),
                ToTensor(expand_dims=False)
            ])
        else:
            return super().weight_transform()


# Make sure to use mode='reflect' when doing RandomRotate otherwise the transform will produce unwanted boundary signal
class LabelToBoundaryElasticTransformer(BaseTransformer):
    def __init__(self, mean, std, phase, label_dtype, **kwargs):
        super().__init__(mean=mean, std=std, phase=phase, label_dtype=label_dtype)
        assert 'angle_spectrum' in kwargs, "'angle_spectrum' argument required"
        self.angle_spectrum = kwargs['angle_spectrum']
        if 'ignore_index' in kwargs:
            self.ignore_index = kwargs['ignore_index']
        else:
            self.ignore_index = None

    def raw_transform(self):
        if self.phase == 'train':
            return Compose([
                Normalize(self.mean, self.std),
                RandomContrast(np.random.RandomState(self.seed)),
                RandomFlip(np.random.RandomState(self.seed)),
                RandomRotate90(np.random.RandomState(self.seed)),
                # rotate in XY only and make sure mode='reflect' is used in order to prevent boundary artifacts
                RandomRotate(np.random.RandomState(self.seed), angle_spectrum=self.angle_spectrum, axes=[(2, 1)],
                             mode='reflect'),
                ElasticDeformation(np.random.RandomState(self.seed), spline_order=3),
                ToTensor(expand_dims=True)
            ])
        else:
            return super().raw_transform()

    def label_transform(self):
        if self.phase == 'train':
            return Compose([
                RandomFlip(np.random.RandomState(self.seed)),
                RandomRotate90(np.random.RandomState(self.seed)),
                # rotate in XY only and make sure mode='reflect' is used in order to prevent boundary artifacts
                RandomRotate(np.random.RandomState(self.seed), angle_spectrum=self.angle_spectrum, axes=[(2, 1)],
                             mode='reflect'),
                # this will give us 4 output channels (one per each offset) with boundary signal
                LabelToBoundary(offsets=(2, 4, 6, 8), ignore_index=self.ignore_index),
                ElasticDeformation(np.random.RandomState(self.seed), spline_order=0),
                ToTensor(expand_dims=False, dtype=self.label_dtype)
            ])
        else:
            return Compose([
                # this will give us 4 output channels (one per each offset) with boundary signal
                LabelToBoundary(offsets=(2, 4, 6, 8), ignore_index=self.ignore_index),
                ToTensor(expand_dims=False, dtype=self.label_dtype)
            ])

    def weight_transform(self):
        if self.phase == 'train':
            return Compose([
                RandomFlip(np.random.RandomState(self.seed)),
                RandomRotate90(np.random.RandomState(self.seed)),
                RandomRotate(np.random.RandomState(self.seed), angle_spectrum=self.angle_spectrum, axes=[(2, 1)],
                             mode='reflect'),
                ElasticDeformation(np.random.RandomState(self.seed), spline_order=3),
                ToTensor(expand_dims=False)
            ])
        else:
            return super().weight_transform()


# Make sure to use mode='reflect' when doing RandomRotate otherwise the transform will produce unwanted boundary signal
class RandomLabelToBoundaryTransformer(BaseTransformer):
    def __init__(self, mean, std, phase, label_dtype, **kwargs):
        super().__init__(mean=mean, std=std, phase=phase, label_dtype=label_dtype)
        assert 'angle_spectrum' in kwargs, "'angle_spectrum' argument required"
        self.angle_spectrum = kwargs['angle_spectrum']
        if 'ignore_index' in kwargs:
            self.ignore_index = kwargs['ignore_index']
        else:
            self.ignore_index = None

    def raw_transform(self):
        if self.phase == 'train':
            return Compose([
                Normalize(self.mean, self.std),
                RandomContrast(np.random.RandomState(self.seed)),
                RandomFlip(np.random.RandomState(self.seed)),
                RandomRotate90(np.random.RandomState(self.seed)),
                # rotate in XY only and make sure mode='reflect' is used in order to prevent boundary artifacts
                RandomRotate(np.random.RandomState(self.seed), angle_spectrum=self.angle_spectrum, axes=[(2, 1)],
                             mode='reflect'),
                ElasticDeformation(np.random.RandomState(self.seed), spline_order=3),
                ToTensor(expand_dims=True)
            ])
        else:
            return super().raw_transform()

    def label_transform(self):
        if self.phase == 'train':
            return Compose([
                RandomFlip(np.random.RandomState(self.seed)),
                RandomRotate90(np.random.RandomState(self.seed)),
                # rotate in XY only and make sure mode='reflect' is used in order to prevent boundary artifacts
                RandomRotate(np.random.RandomState(self.seed), angle_spectrum=self.angle_spectrum, axes=[(2, 1)],
                             mode='reflect'),
                RandomLabelToBoundary(np.random.RandomState(self.seed), ignore_index=self.ignore_index),
                ElasticDeformation(np.random.RandomState(self.seed), spline_order=0),
                ToTensor(expand_dims=True, dtype=self.label_dtype)
            ])
        else:
            return Compose([
                RandomLabelToBoundary(np.random.RandomState(self.seed), ignore_index=self.ignore_index),
                # this will give us 6 output channels with boundary signal
                ToTensor(expand_dims=True, dtype=self.label_dtype)
            ])

    def weight_transform(self):
        if self.phase == 'train':
            return Compose([
                RandomFlip(np.random.RandomState(self.seed)),
                RandomRotate90(np.random.RandomState(self.seed)),
                RandomRotate(np.random.RandomState(self.seed), angle_spectrum=self.angle_spectrum, axes=(2, 0),
                             mode='reflect'),
                ElasticDeformation(np.random.RandomState(self.seed), spline_order=3),
                ToTensor(expand_dims=False)
            ])
        else:
            return super().weight_transform()


# Use this transformer if you want to validate your model with AveragePrecision metric. Returns boundary signal during
# training and original ground truth during validation
class LabelToBoundaryAPTransformer(BaseTransformer):
    def __init__(self, mean, std, phase, label_dtype, **kwargs):
        super().__init__(mean=mean, std=std, phase=phase, label_dtype=label_dtype)
        assert 'angle_spectrum' in kwargs, "'angle_spectrum' argument required"
        self.angle_spectrum = kwargs['angle_spectrum']
        if 'ignore_index' in kwargs:
            self.ignore_index = kwargs['ignore_index']
        else:
            self.ignore_index = None

    def raw_transform(self):
        if self.phase == 'train':
            return Compose([
                Normalize(self.mean, self.std),
                RandomContrast(np.random.RandomState(self.seed)),
                RandomFlip(np.random.RandomState(self.seed)),
                RandomRotate90(np.random.RandomState(self.seed)),
                # rotate in XY only and make sure mode='reflect' is used in order to prevent boundary artifacts
                RandomRotate(np.random.RandomState(self.seed), angle_spectrum=self.angle_spectrum, axes=[(2, 1)],
                             mode='reflect'),
                ToTensor(expand_dims=True)
            ])
        else:
            return super().raw_transform()

    def label_transform(self):
        if self.phase == 'train':
            return Compose([
                RandomFlip(np.random.RandomState(self.seed)),
                RandomRotate90(np.random.RandomState(self.seed)),
                # rotate in XY only and make sure mode='reflect' is used in order to prevent boundary artifacts
                RandomRotate(np.random.RandomState(self.seed), angle_spectrum=self.angle_spectrum, axes=[(2, 1)],
                             mode='reflect'),
                # this will give us 4 output channels with boundary signal
                LabelToBoundary(offsets=(2, 4, 6, 8), ignore_index=self.ignore_index),
                ToTensor(expand_dims=False, dtype=self.label_dtype)
            ])
        else:
            return Compose([
                # just return the ground truth
                ToTensor(expand_dims=False, dtype=self.label_dtype)
            ])

    def weight_transform(self):
        if self.phase == 'train':
            return Compose([
                RandomFlip(np.random.RandomState(self.seed)),
                RandomRotate90(np.random.RandomState(self.seed)),
                RandomRotate(np.random.RandomState(self.seed), angle_spectrum=self.angle_spectrum, axes=[(2, 1)],
                             mode='reflect'),
                ToTensor(expand_dims=False)
            ])
        else:
            return super().weight_transform()


class My_transformer(BaseTransformer):

    def __init__(self, mean, std, phase, label_dtype, **kwargs):
        super().__init__(mean=mean, std=std, phase=phase, label_dtype=label_dtype)
        assert 'clip_min' in kwargs, "'clip_min' argument required"
        assert 'clip_max' in kwargs, "'clip_max' argument required"
        self.clip_min = kwargs['clip_min']
        self.clip_max = kwargs['clip_max']
        assert 'angle_spectrum' in kwargs, "'angle_spectrum' argument required"
        self.angle_spectrum = kwargs['angle_spectrum']

    def raw_transform(self):
        if self.phase == 'train':
            return Compose([
                Clip(self.clip_min, self.clip_max),
                # Normalize(self.mean, self.std),
                NormalizeV2(),
                RandomFlip(np.random.RandomState(self.seed), axes=(2, 0)),
                RandomRotate90(np.random.RandomState(self.seed), axes=(2, 0)),
                # RandomRotate(np.random.RandomState(self.seed), angle_spectrum=self.angle_spectrum, axes=[(2, 0)], cval=0),
                # ElasticDeformation(np.random.RandomState(self.seed), 0),
                # RandomContrast(np.random.RandomState(self.seed)),
                # ToTensor(expand_dims=True)
            ])
        else:
            return Compose([
                Clip(self.clip_min, self.clip_max),
                # Normalize(self.mean, self.std),
                NormalizeV2()
                # ToTensor(expand_dims=True)
            ])

    def label_transform(self):
        if self.phase == 'train':
            return Compose([
                RandomFlip(np.random.RandomState(self.seed), axes=(2, 0)),
                # RandomRotate(np.random.RandomState(self.seed), angle_spectrum=self.angle_spectrum, axes=[(2, 0)], cval=0),
                RandomRotate90(np.random.RandomState(self.seed),axes=(2,0)),
                # ElasticDeformation(np.random.RandomState(self.seed), 0),
                # ToTensor(expand_dims=False, dtype=self.label_dtype)
            ])
        else:
            return Compose([
                Identity()
            ])
            # return super().label_transform()
