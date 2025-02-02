import os
import random
import torch
import torch.utils.data as data
from PIL import Image
from torchvision import transforms

import numpy as np


class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (List[Transform]): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class Scale(object):
    """
    Rescale the input PIL.Image to given size.
    """

    def __init__(self, size):
        super(Scale, self).__init__()
        self.size = (size, size)

    def _scale(self, img, interpolation=Image.BILINEAR):
        return img.resize(self.size, interpolation)

    def __call__(self, input):
        input['img'] = self._scale(input['img'])
        input['gt'] = self._scale(input['gt'])
        return input


class Random_Crop(object):
    def __init__(self, t_size):
        self.t_size = t_size

    def _crop(self, img, x1, y1, x2, y2):
        return img.crop((x1, y1, x2, y2))

    def __call__(self, input):
        img = input['img']
        w, h = img.size

        if w != self.t_size and h != self.t_size:
            x1 = random.randint(0, w - self.t_size)
            y1 = random.randint(0, h - self.t_size)
            input['img'] = self._crop(img, x1, y1, x1 + self.t_size, y1 + self.t_size)
            input['gt'] = self._crop(input['gt'], x1, y1, x1 + self.t_size, y1 + self.t_size)

        return input


class Random_Flip(object):
    def _flip(self, img):
        return img.transpose(Image.FLIP_LEFT_RIGHT)

    def __call__(self, input):
        if random.random() < 0.5:
           input['img'] = self._flip(input['img'])
           input['gt'] = self._flip(input['gt'])

        return input


class normalization(object):
    def __init__(self, split, scale_size=None):
        self.split = split
        if self.split == 'train':
            self.img_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225]
                    )
                ]
            )
            self.gt_transform = transforms.ToTensor()
        elif self.split == 'test':
            self.img_transform = transforms.Compose(
                [
                    transforms.Resize((scale_size, scale_size), interpolation=Image.BILINEAR),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225]
                    )
                ]
            )
        else:
            raise Exception("split not recognized")

    def __call__(self, input):
        if self.split == 'train':
            input['img'] = self.img_transform(input['img'])
            input['gt'] = self.gt_transform(input['gt'])
        elif self.split == 'test':
            input = self.img_transform(input)
        return input


dataset_dirs = {
    "DC": "./dataset/train_data/DUTS_class/img",
    "C9": "./dataset/train_data/CoCo9k/img",
    "CS": "./dataset/train_data/CoCoSeg/img",
    "OWDC": "./dataset/train_data/OWDUTS_class/img",
    "OWCS": "./dataset/train_data/OWCoCo_Seg/img",
}


class CoSOD_Train(data.Dataset):
    def __init__(self, args, split='train'):
        self.split = split

        self.train_data_set = args.train_data_set.split('+')

        if args.model == "DMT+O":
            self.train_data_set = ['OWDC', 'OWCS']

        self.all_imgs_dirs_list, self.all_gts_dirs_list, self.data_flag = [], [], []
        for dataset in self.train_data_set:
            imgs_dirs_list, gts_dirs_list = self.get_imgs_gts_dirs(dataset_dirs[dataset])
            if "DC" in self.train_data_set or "OWDC" in self.train_data_set:
                if dataset == "DC" or dataset == "OWDC":
                    data_flag = [True for i in range(len(imgs_dirs_list))]
                else:
                    data_flag = [False for i in range(len(imgs_dirs_list))]
            elif "C9" in self.train_data_set and dataset == "C9":
                data_flag = [True for i in range(len(imgs_dirs_list))]
            else:
                data_flag = [False for i in range(len(imgs_dirs_list))]
            self.all_imgs_dirs_list += imgs_dirs_list
            self.all_gts_dirs_list += gts_dirs_list
            self.data_flag += data_flag

        inds = [i for i in range(len(self.all_imgs_dirs_list))]
        np.random.shuffle(inds)

        self.all_imgs_dirs_list = [self.all_imgs_dirs_list[i] for i in inds]
        self.all_gts_dirs_list = [self.all_gts_dirs_list[i] for i in inds]
        self.data_flag = [self.data_flag[i] for i in inds]

        self.max_num = args.max_num

        self.size = args.img_size
        self.scale_size = args.scale_size

        if "DC" in self.train_data_set:
            self.syn_root = '/'.join(dataset_dirs["DC"].split('/')[:-1]) + "_syn"
        elif "OWDC" in self.train_data_set:
            self.syn_root = '/'.join(dataset_dirs["DC"].split('/')[:-1]) + "_syn"
        elif "C9" in self.train_data_set:
            self.syn_root = '/'.join(dataset_dirs["C9"].split('/')[:-1]) + "_syn"

        self._augmentation()

    def get_imgs_gts_dirs(self, root):
        class_names = os.listdir(root)
        classes_dir = list(
            map(lambda class_name: os.path.join(root, class_name), class_names)
        )
        imgs_names_list = [
            os.listdir(idir) for idir in classes_dir
        ]
        imgs_dirs_list = [
            list(
                map(lambda img_name: os.path.join(classes_dir[idx], img_name),
                    imgs_names_list[idx])
            )
            for idx in range(len(classes_dir))
        ]
        if 'CoCo_Seg' in root:
            gts_dirs_list = [
                [img.replace('img', 'gt') for img in sublist]
                for sublist in imgs_dirs_list
            ]
        else:
            gts_dirs_list = [
                [img.replace('img', 'gt').replace('jpg', 'png') for img in sublist]
                for sublist in imgs_dirs_list
            ]
        return imgs_dirs_list, gts_dirs_list

    def _augmentation(self):
        if self.split == 'train':
            self.joint_transform = Compose([
                Scale(self.scale_size),
                Random_Crop(self.size),
                Random_Flip(),
            ])
        elif self.split == 'test':
            self.joint_transform = None
        else:
            raise Exception("split not recognized")
        self.normalization = normalization(self.split, self.size)

    def __getitem__(self, item):
        imgs_path = self.all_imgs_dirs_list[item]
        gts_path = self.all_gts_dirs_list[item]

        flag = self.data_flag[item]

        num = len(imgs_path)
        if num > self.max_num:
            sample_list = random.sample(range(num), self.max_num)
            imgs_path = [imgs_path[i] for i in sample_list]
            gts_path = [gts_path[i] for i in sample_list]
            num = self.max_num

        imgs = torch.zeros(num, 3, self.size, self.size)
        gts = torch.zeros(num, 1, self.size, self.size)

        ori_sizes = []

        for idx in range(num):
            if flag:
                # data from our dataset
                # random replace to syn img or do not replace
                select_num = random.randint(1, 5)
                if select_num == 4:
                    # select original img
                    img_path = imgs_path[idx]
                    gt_path = gts_path[idx]
                if 1 <= select_num <= 3:
                    # select syn img
                    imgs_path_split = imgs_path[idx].split('/')
                    class_name, img_name = imgs_path_split[-2], imgs_path_split[-1]
                    syn_img_name = img_name[:-4] + '_syn' + str(select_num) + '.png'
                    syn_naive_dir = os.path.join(self.syn_root, "naive")
                    img_path = os.path.join(syn_naive_dir, "img", class_name, syn_img_name)
                    if not os.path.exists(img_path):
                        img_path = imgs_path[idx]
                    gt_path = gts_path[idx]
                if select_num == 5:
                    # select reverse syn img
                    select_reverse_num = random.randint(1, 3)
                    imgs_path_split = imgs_path[idx].split('/')
                    class_name, img_name = imgs_path_split[-2], imgs_path_split[-1]
                    rev_syn_img_name = img_name[:-4]+'_ReverseSyn'+str(select_reverse_num)+'.png'
                    syn_reverse_dir = os.path.join(self.syn_root, "reverse")
                    img_path = os.path.join(syn_reverse_dir, "img", class_name, rev_syn_img_name)
                    gt_path = os.path.join(syn_reverse_dir, "gt", class_name, rev_syn_img_name)
                    if not os.path.exists(img_path):
                        img_path = imgs_path[idx]
                        gt_path = gts_path[idx]
            else:
                # data from coco
                img_path = imgs_path[idx]
                gt_path = gts_path[idx]

            zip_data = {}

            img = Image.open(img_path).convert('RGB')
            gt = Image.open(gt_path).convert('L')

            # print(img_path, co_gt_path, sal_gt_path)

            ori_sizes.append((img.size[1], img.size[0]))

            zip_data['img'] = img
            zip_data['gt'] = gt

            zip_data = self.joint_transform(zip_data)
            zip_data = self.normalization(zip_data)

            imgs[idx] = zip_data['img']
            gts[idx] = zip_data['gt']

        return {
            "imgs": imgs,
            "gts": gts,
        }

    def __len__(self):
        return len(self.all_imgs_dirs_list)


class CoData_Test(data.Dataset):
    def __init__(self, img_root, img_size):
        class_list = os.listdir(os.path.join(img_root, 'Image'))
        self.transform = normalization(split='test', scale_size=img_size)
        self.classes_dirs_list = list(
            map(lambda x: os.path.join(img_root, 'Image', x), class_list)
        )
        self.sizes = [img_size, img_size]

    def __getitem__(self, item):
        class_dir = self.classes_dirs_list[item]
        img_names = os.listdir(class_dir)
        num = len(img_names)
        img_paths = list(
            map(lambda x: os.path.join(class_dir, x), img_names)
        )

        imgs = torch.zeros(num, 3, self.sizes[0], self.sizes[1])

        subpaths = []
        ori_sizes = []

        for idx in range(num):
            img = Image.open(img_paths[idx]).convert('RGB')
            img_path_split = img_paths[idx].split('/')
            subpaths.append(
                os.path.join(
                    img_path_split[-2],
                    img_path_split[-1][:-4]+'.png')
            )
            ori_sizes.append((img.size[1], img.size[0]))
            img = self.transform(img)
            imgs[idx] = img

        return {
            "imgs": imgs,
            "subpaths": subpaths,
            "ori_sizes": ori_sizes
        }

    def __len__(self):
        return len(self.classes_dirs_list)


def build_data_loader(args, mode):
    '''
    :param args: arg parser object for strategy
    :param mode: training or testing
    :return: data iterator
    '''
    if mode == "train":
        train_dataset = CoSOD_Train(args, 'train')
        data_loader = data.DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
        )
        return data_loader
    elif mode == "test":
        test_root_dir = args.test_data_root
        test_datasets = args.test_datasets
        data_loaders = {}
        for dataset in test_datasets:
            data_root = os.path.join(test_root_dir, dataset)
            test_dataset = CoData_Test(
                img_root=data_root,
                img_size=args.img_size
            )
            data_loader = data.DataLoader(
                dataset=test_dataset,
                batch_size=args.batch_size
            )
            data_loaders[dataset] = data_loader
        return data_loaders
    else:
        raise RuntimeError
