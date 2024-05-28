from torch.utils import data as data
from torchvision.transforms.functional import normalize
from os import path as osp
from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb, paired_paths_from_meta_info_file
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, bgr2ycbcr, imfrombytes, img2tensor,scandir
from basicsr.utils.registry import DATASET_REGISTRY
from os import listdir
import cv2
import scipy.io as sio
import torch

def tri_paths_from_folder(folders, keys, filename_tmpl):
    """Generate paired paths from folders.

    Args:
        folders (list[str]): A list of folder path. The order of list should
            be [input_folder, gt_folder].
        keys (list[str]): A list of keys identifying folders. The order should
            be in consistent with folders, e.g., ['lq', 'gt'].
        filename_tmpl (str): Template for each filename. Note that the
            template excludes the file extension. Usually the filename_tmpl is
            for files in the input folder.

    Returns:
        list[str]: Returned path list.
    """

    assert len(keys) == 3, f'The len of keys should be 3 with [input_key, gt_key, tilt_key]. But got {len(keys)}'
    input_key, gt_key, tilt_key = keys

    # input_paths = list(scandir(input_folder))
    # gt_paths = list(scandir(gt_folder))
    # assert len(input_paths) == len(gt_paths), (f'{input_key} and {gt_key} datasets have different number of images: '
    #                                            f'{len(input_paths)}, {len(gt_paths)}.')
    paths = []
    for folder in listdir(folders):
        gt_path=osp.join(folders, folder, 'gt.jpg')
        input_folder=osp.join(folders,folder,'turb')
        input_paths = list(scandir(input_folder))
        tilt_folder=osp.join(folders,folder,'tilt')
        tilt_paths = list(scandir(tilt_folder))
        # assert len(input_paths) == len(tilt_paths), (f'{input_key} and {tilt_key} datasets have different number of images: '
        #                                            f'{len(input_paths)}, {len(tilt_paths)}.')
        for input_name in input_paths:
            basename, ext = osp.splitext(osp.basename(input_name))
            tilt_name = f'{filename_tmpl.format(basename)}{ext}'
            tilt_flow_field_name = f'{filename_tmpl.format(basename)}.mat'
            input_path = osp.join(input_folder, input_name)
            tilt_path = osp.join(tilt_folder,tilt_name)
            tilt_flow_field_path = osp.join(tilt_folder,tilt_flow_field_name)
            assert tilt_name in tilt_paths, f'{tilt_name} is not in {tilt_key}_paths.'
            paths.append(dict([(f'{input_key}_path', input_path), (f'{gt_key}_path', gt_path), (f'{tilt_key}_path', tilt_path),(f'{tilt_key}_field_path', tilt_flow_field_path)]))
    return paths
@DATASET_REGISTRY.register()
class TurbImageDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and GT image pairs.

    There are three modes:

    1. **lmdb**: Use lmdb files. If opt['io_backend'] == lmdb.
    2. **meta_info_file**: Use meta information file to generate paths. \
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. **folder**: Scan folders to generate paths. The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
        dataroot_gt (str): Data root path for gt.
        dataroot_lq (str): Data root path for lq.
        meta_info_file (str): Path for meta information file.
        io_backend (dict): IO backend type and other kwarg.
        filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
            Default: '{}'.
        gt_size (int): Cropped patched size for gt patches.
        use_hflip (bool): Use horizontal flips.
        use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
        scale (bool): Scale, which will be added automatically.
        phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(TurbImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.turb_size= opt['turb_size']
        self.gt_size= opt['gt_size']

        self.train_folder= opt['dataroot']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'


        self.paths = tri_paths_from_folder(self.train_folder, ['turb', 'gt','tilt'], self.filename_tmpl)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)
        img_gt=cv2.resize(img_gt,(self.gt_size,self.gt_size))

        turb_path = self.paths[index]['turb_path']
        img_bytes = self.file_client.get(turb_path, 'turb')
        img_turb = imfrombytes(img_bytes, float32=True)
        img_turb=cv2.resize(img_turb,(self.turb_size,self.turb_size))

        tilt_path = self.paths[index]['tilt_path']
        img_bytes = self.file_client.get(tilt_path, 'tilt')
        img_tilt = imfrombytes(img_bytes, float32=True)
        img_tilt=cv2.resize(img_tilt,(self.turb_size,self.turb_size))


        # # augmentation for training
        # if self.opt['phase'] == 'train':
        #     gt_size = self.opt['gt_size']
        #     # random crop
        #     img_gt, img_turb = paired_random_crop(img_gt, img_turb, gt_size, scale, gt_path)
        #     # flip, rotation
        #     img_gt, img_turb = augment([img_gt, img_turb], self.opt['use_hflip'], self.opt['use_rot'])

        # color space transform
        if 'color' in self.opt and self.opt['color'] == 'y':
            img_gt = bgr2ycbcr(img_gt, y_only=True)[..., None]
            img_turb = bgr2ycbcr(img_turb, y_only=True)[..., None]
            img_tilt = bgr2ycbcr(img_tilt, y_only=True)[..., None]

        # crop the unmatched GT images during validation or testing, especially for SR benchmark datasets
        # TODO: It is better to update the datasets, rather than force to crop
        # if self.opt['phase'] != 'train':
        #     img_gt = img_gt[0:img_turb.shape[0] * scale, 0:img_turb.shape[1] * scale, :]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_turb,img_tilt = img2tensor([img_gt, img_turb, img_tilt], bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_turb, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        if osp.exists(self.paths[index]['tilt_field_path']):
            tilt_field=torch.from_numpy(sio.loadmat(self.paths[index]['tilt_field_path'])['D'])
            return {'turb': img_turb, 'gt': img_gt, 'tilt': img_tilt, 'turb_path': turb_path, 'gt_path': gt_path, 'tilt_path': tilt_path,'tilt_field':tilt_field}
        else:
            return {'turb': img_turb, 'gt': img_gt, 'tilt': img_tilt, 'turb_path': turb_path, 'gt_path': gt_path, 'tilt_path': tilt_path}

    def __len__(self):
        return len(self.paths)
