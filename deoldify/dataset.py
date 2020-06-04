import fastai
from fastai import *
from fastai.core import *
from fastai.vision.transform import get_transforms
from fastai.vision.data import ImageImageList, ImageDataBunch, imagenet_stats
from fastai.vision import *
from .augs import noisify
import os.path as osp
import cv2
from .utils import *
import torchvision.transforms as transforms
import glob
import torch.utils.data as data

def get_colorize_data(
    sz: int,
    bs: int,
    crappy_path: Path,
    good_path: Path,
    random_seed: int = None,
    keep_pct: float = 1.0,
    num_workers: int = 8,
    stats: tuple = imagenet_stats,
    xtra_tfms=[],
) -> ImageDataBunch:
    
    src = (
        ImageImageList.from_folder(crappy_path, convert_mode='RGB')
        .use_partial_data(sample_pct=keep_pct, seed=random_seed)
        .split_by_rand_pct(0.1, seed=random_seed)
    )

    data = (
        src.label_from_func(lambda x: good_path / x.relative_to(crappy_path))
        .transform(
            get_transforms(
                max_zoom=1.2, max_lighting=0.5, max_warp=0.25, xtra_tfms=xtra_tfms
            ),
            size=sz,
            tfm_y=True,
        )
        .databunch(bs=bs, num_workers=num_workers, no_check=True)
        .normalize(stats, do_y=True)
    )

    data.c = 3
    return data


def get_dummy_databunch() -> ImageDataBunch:
    path = Path('./dummy/')
    return get_colorize_data(
        sz=1, bs=1, crappy_path=path, good_path=path, keep_pct=0.001
    )


class AutopaintSrcItem(ItemBase):
    def __init__(self,gt_path,sk_path,img_size,_is_train):
        gt = cv2.imread(gt_path)
        sketch = cv2.imread(sk_path,0)
        sketch = cv2.fastNlMeansDenoising(sketch,10,10,7,21)
        if _is_train:
            desensitize_level = random.randint(10,100)
            sketch = sensitive(sketch,s=desensitize_level)
            if random.random()<0.2:
                sketch = cv2.GaussianBlur(sketch,(3,3),0)

        if _is_train:
            point_map = generate_user_point(gt,img_size)
        else:
            sketch = my_resize(sketch,img_size)
            gt = my_resize(gt,img_size)
            point_map = generate_user_point(gt,is_random=False)

        sketch = mini_norm(sketch)
        sketch = Image.fromarray(sketch).convert('L')


        sketch = transforms.ToTensor()(sketch)
        point_map = transforms.ToTensor()(point_map)
        self.sketch = sketch
        self.point_map = point_map
        self.obj = (sketch,point_map)

        input = torch.cat((sketch,point_map),0)
        input = transforms.Normalize(mean=(0.5,0.5,0.5,0.5,0.5), std=(0.5,0.5,0.5,0.5,0.5))(input)
        self.data = input
    def apply_tfms(self,tfms, **kwargs):
        self.sketch = self.sketch.apply_tfms(tfms, **kwargs)
        self.point_map = self.point_map.apply_tfms(tfms, **kwargs)
        input = torch.cat((self.sketch,self.point_map),0)
        input = transforms.Normalize(mean=(0.5,0.5,0.5,0.5,0.5), std=(0.5,0.5,0.5,0.5,0.5))(input)
        self.data = input
        return self
    def to_one(self):
        color_sketch = torch.cat(3*[self.sketch])
        color_point_map = self.point_map[:3,:,:]
        return Image(0.5 + torch.cat([color_sketch,color_point_map], 2) / 2)



class AlacDataSet(data.Dataset):
    def __init__(self, data_path, sketch_path, img_size,is_train):
        self._data_path = data_path
        self._sketch_path = sketch_path
        self._is_train = is_train
        self.points_generator = user_point_generator()
        self.img_size = img_size

        self.transform_gt = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
        self.transform_point = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5,0.5), std=(0.5, 0.5, 0.5,0.5))])
        self.transform_sketch = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])])

        self.gt_img =glob.glob(self._data_path+'/*')

        print('length of dataset is {}'.format(len(self.gt_img)))

    def __getitem__(self, index):

        gt_path = self.gt_img[index]
        pnames = gt_path.split('/')
        sk_path = osp.join(self._sketch_path,pnames[-1])

        gt = cv2.imread(gt_path)
        sketch = cv2.imread(sk_path,0)
        sketch = cv2.fastNlMeansDenoising(sketch,10,10,7,21)
        if self._is_train:
            h,w = sketch.shape
            if h>w:
                h_start = random.randint(0,h-w-1)
                gt = gt[h_start:h_start+w,:]
                sketch = sketch[h_start:h_start+w,:]
            if w>h:
                w_start = random.randint(0,w-h-1)
                gt = gt[:,w_start:w_start+h]
                sketch = sketch[:,w_start:w_start+h]
            desensitize_level = random.randint(10,100)
            sketch = sensitive(sketch,s=desensitize_level)
            #random flip
            if random.random()<0.5:
                sketch = cv2.flip(sketch,1)
                gt = cv2.flip(gt,1)
            #random center crop
            if random.random()<0.5:
                crop_value = random.randint(1,50)
                sketch = sketch[crop_value:-crop_value,crop_value:-crop_value]
                gt = gt[crop_value:-crop_value,crop_value:-crop_value]
            if random.random()<0.2:
                sketch = cv2.GaussianBlur(sketch,(3,3),0)

        if self._is_train:
            sketch = cv2.resize(sketch,(self.img_size,self.img_size),interpolation=cv2.INTER_LANCZOS4)
            gt = cv2.resize(gt,(self.img_size,self.img_size),interpolation=cv2.INTER_LANCZOS4)
            point_map = self.points_generator.generate(gt,self.img_size)
        else:
            sketch = my_resize(sketch,self.img_size)
            gt = my_resize(gt,self.img_size)
            point_map = self.points_generator.generate(gt,is_random=False)

        sketch = mini_norm(sketch)
        sketch = Image.fromarray(sketch).convert('L')
        gt = Image.fromarray(cv2.cvtColor(gt,cv2.COLOR_BGR2RGB)).convert('RGB')

        sketch = transforms.ToTensor()(sketch)
        point_map = transforms.ToTensor()(point_map)
        src = torch.cat((sketch,point_map),0)
        src = transforms.Normalize(mean=(0.5, 0.5, 0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5, 0.5, 0.5))(src)
        gt = self.transform_gt(gt)

        return (src,gt)

    def __len__(self):
        return len(self.gt_img)
