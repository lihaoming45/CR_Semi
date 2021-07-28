import os
import glob
import random
# from albumentations.augmentations.transforms import Rotate
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch
import nibabel as nib
import SimpleITK as sitk
import numpy as np
import cv2 as cv
import transforms as transforms
from hdf5 import SliceBuilder
import random
from resize_3D import itk_resample
# from utils import foll_count, edge_mask


class MySet(Dataset):
    def __init__(self, train_dir, num_class=3, aug_decision=False, phase='train', slice_builder_cls=SliceBuilder,
                 patch_shape=(144, 112, 128), stride_shape=(96, 56, 64), new_shape=(224, 224)):
        self.data_dir = train_dir['data']
        self.label_dir = train_dir['label']

        self.num_class = num_class
        self.aug_decision = aug_decision
        self.transformer_builder = transforms.TransformerBuilder(transforms.My_transformer,
                                                                 {'label_dtype': 'float32', 'angle_spectrum': 30,
                                                                  'clip_min': 10, 'clip_max': 240})
        self.transformer_builder.phase = phase
        self.phase = phase
        self.data, self.label, self.voxel_spacing = self.resize_3D(self.data_dir, self.label_dir, new_shape=new_shape)

        if self.label is not None:
            self.label[self.label == 128] = 1
            self.label[self.label == 255] = 2
            # self.label[self.label==3]=0
            # self.label[self.label==1]=4
            # self.label = self.label/2
            self.label = self.label.astype(np.uint8)
            # self.edge_label = edge_mask(self.label)

        if aug_decision and self.label is not None:
            self.data, self.label = self.transform_pro(trans_builder=self.transformer_builder, data=self.data,
                                                       label=self.label)
        else:
            self.data = self.normalize2(self.data)



        slice_builder = slice_builder_cls(self.data, self.label, patch_shape=patch_shape, stride_shape=stride_shape)
        self.raw_slice = slice_builder.raw_slices
        self.label_slice = slice_builder.label_slices

        self.patch_count = len(self.raw_slice)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise StopIteration

        raw_idx = self.raw_slice[idx]
        data = self.data[raw_idx]
        if self.label_slice is not None:
            label_idx = self.label_slice[idx]
            mask = self.label[label_idx]

            mask_itk = sitk.GetImageFromArray(mask)
            newshape = (int(mask.shape[0] / 4), int(mask.shape[1] / 4), int(mask.shape[2] / 4))
            mask_CR = itk_resample(mask_itk, new_shape=newshape, islabel=True)
            mask_CR = sitk.GetArrayFromImage(mask_CR)
        else:
            mask = None

        # numFoll_class = foll_count(mask)
        # edge_mask=self.edge_label[label_idx]

        # data = ndarray_nearest_neighbour_scaling(data, 224, 128)
        # mask = ndarray_nearest_neighbour_scaling(mask, 224, 128)

        # if self.aug_decision == False:
        #     data = self.normalize2(data)

        data = data.astype(np.float32)
        data = np.expand_dims(data, axis=0)  # 输入channel=1
        data_tensor = torch.from_numpy(data)
        if mask is not None:
            mask = mask.astype(np.uint8)
            mask_tensor = torch.from_numpy(mask)

            mask_CR =mask_CR.astype(np.uint8)
            mask_CR = torch.from_numpy(mask_CR)


        else:
            mask_tensor = 0
            self.label = 0
        # numFoll_tensor = torch.from_numpy(np.array(numFoll_class))

        if self.phase == 'train':
            return data_tensor, mask_tensor,mask_CR, idx
        else:
            # return data_tensor, mask_tensor, idx
            return data_tensor, mask_tensor, idx, self.patch_count, self.label, self.data.shape, self.voxel_spacing

    @staticmethod
    def normalize(data):
        data = data.astype(np.float32)
        data = (data - np.min(data)) / (np.max(data) - np.min(data))
        # data = data / 255
        # data = data*255
        return data

    @staticmethod
    def normalize2(data):
        data = data.astype(np.float32)
        data = (data-np.mean(data))/(np.std(data))
        return data

    @staticmethod
    def resize_3D(data_dir, label_dir, new_shape=(256, 256)):
        data = sitk.ReadImage(data_dir)
        ora_shape = (data.GetSize()[2],data.GetSize()[1],data.GetSize()[0])

        if label_dir is not None:
            label = sitk.ReadImage(label_dir)
            label = itk_resample(label, new_shape=new_shape, islabel=True)
            label = sitk.GetArrayFromImage(label)

        else:
            label = None
        data = itk_resample(data, new_shape=new_shape, islabel=False)
        data = sitk.GetArrayFromImage(data)

        scale = np.array(ora_shape) / np.array(data.shape)
        voxel_spacing = scale * 0.2

        # data = np.transpose(data,axes=(2,1,0))
        # label = np.transpose(label,axes=(2,1,0))

        return data, label, voxel_spacing

    @staticmethod
    def adjustlabel(label, num_class):
        # gt_lsit = list(np.unique(label))
        gt_dict = {0: 0, 128: 1, 255: 2}
        # new_mask = np.zeros((num_class,) + label.shape)
        for key in gt_dict:
            label[label == key] = gt_dict[key]
        return label

    @staticmethod
    def calculate_mean_std(data):
        return data[...].mean(keepdims=True), data[...].std(keepdims=True)

    def transform_pro(self, trans_builder, data, label):
        mean, std = self.calculate_mean_std(data)
        trans_builder.mean = mean
        trans_builder.std = std
        trans_builder.phase = self.phase
        transformer = trans_builder.build()

        data_transformer = transformer.raw_transform()
        label_transformer = transformer.label_transform()
        data = data_transformer(data)
        label = label_transformer(label)

        return data, label

    def __len__(self):
        return self.patch_count  # 返回样本数


def create_list_challengeData(data_path, ratio=0.8):
    data_list = glob.glob(os.path.join(data_path, '*/data.nii.gz'))
    label_list = glob.glob(os.path.join(data_path, '*/label.nii.gz'))

    # label_name = 'label.nii.gz'
    # data_name = 'data.nii.gz'

    data_list.sort(key=lambda x: int(x.split('/')[-2]))
    label_list.sort(key=lambda x: int(x.split('/')[-2]))
    list_all = [{'data': os.path.join(path_data), 'label': os.path.join(path_label)} for path_data, path_label in
                zip(data_list, label_list)]

    cut = int(ratio * len(list_all))
    train_list = list_all[:cut]
    test_list = list_all[cut:]

    # random.shuffle(train_list)

    return train_list, test_list


def create_list_MYGE(data_path, ratio=0.9, split_list=['MY', 'GE'], data_folder='synthesis_resize',
                     label_folder='label_resize',crop_vaild =False,is_small=False):
    all_train_list = []
    all_test_list = []


    for split in split_list:

        data_list = glob.glob(os.path.join(data_path, split, data_folder, '*.nii.gz'))
        data_list.sort(key=lambda x: x.split('/')[-1])
        if label_folder is not None:
            label_list = glob.glob(os.path.join(data_path, split, label_folder, '*.nii.gz'))
            label_list.sort(key=lambda x: x.split('/')[-1])

            list_all = [{'data': os.path.join(path_data), 'label': os.path.join(path_label)} for path_data, path_label
                        in
                        zip(data_list, label_list)]
        else:
            list_all = [{'data': os.path.join(path_data), 'label': None} for path_data in data_list]

        if crop_vaild:
            cut = int((1-ratio) * len(list_all))
            test_list = list_all[:(cut+1)]
            train_list = list_all[(cut+1):]
            all_train_list.extend(train_list)
            all_test_list.extend(test_list)

        else:
            cut = int(ratio * len(list_all))
            train_list = list_all[:cut]
            test_list = list_all[cut:]
            all_train_list.extend(train_list)
            all_test_list.extend(test_list)


        # random.shuffle(all_train_list)
        # random.shuffle(all_test_list)

    # random.shuffle(train_list)
    if is_small:
        return all_train_list[10:30], all_test_list

    return all_train_list, all_test_list


def train_val_loader(train_list, val_list, patch_shape, stride_shape, new_shape, aug_decision):
    train_datasets = []
    for train_dir in train_list:
        train_dataset = MySet(train_dir, aug_decision=aug_decision, phase='train', patch_shape=patch_shape,
                              stride_shape=stride_shape, new_shape=new_shape)

        train_datasets.append(train_dataset)
    concat_train = ConcatDataset(train_datasets)

    if val_list is not None:
        val_datasets = []
        for val_dir in val_list:
            val_dataset = MySet(val_dir, aug_decision=aug_decision, phase='val', patch_shape=patch_shape,
                                stride_shape=stride_shape, new_shape=new_shape)

            val_datasets.append(val_dataset)


        concat_val = ConcatDataset(val_datasets)

        return concat_train, concat_val
    else:
        return concat_train

def val_loader(val_list, patch_shape, stride_shape, new_shape, aug_decision):
    val_datasets = []
    for val_dir in val_list:
        val_dataset = MySet(val_dir, aug_decision=aug_decision, phase='val', patch_shape=patch_shape,
                            stride_shape=stride_shape, new_shape=new_shape)

        val_datasets.append(val_dataset)

    concat_val = ConcatDataset(val_datasets)
    return concat_val

# if __name__ == "__main__":
#     all_train_list = []
#     all_test_list = []
#     base_path = "/home/lhm/All_data/formal_data_3D(MY_GE)"
#     train_list, test_list = create_list_MYGE(base_path)
#     train_set, test_set = train_val_loader(train_list, test_list, patch_shape=(144, 112, 128),
#                                            stride_shape=(96, 56, 64), aug_decision=False)
#
#     # all_train_list = MY_train_list + GE_train_list
#     # all_test_list = MY_test_list + GE_test_list
#     random.shuffle(all_train_list)
#     random.shuffle(all_train_list)
