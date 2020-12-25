from os.path import join
import random
import numpy as np
from scipy.io import loadmat
import torch.utils.data as data
import util.util_img


class Dataset(data.Dataset):
    data_root = 'Backup_Drive/shapenet'
    list_root = join(data_root, 'status')
    status_and_suffix = {
        'rgb': {
            'status': 'rgb.txt',
            'suffix': '_rgb.png',
        },
        'silhou': {
            'status': 'silhou.txt',
            'suffix': '_silhouette.png',
        },
        'voxel': {
            'status': 'vox_rot.txt',
            'suffix': '_gt_rotvox_samescale_128.npz'
        }}

	
    class_aliases = {
        'chair': '03001627',
        'table': '04379243',
        'plane': '02691156',
        'car': '02958343',
        'all': '03001627+04379243+02691156+02958343',
        }
		
    class_list = class_aliases['all'].split('+')

	
    def add_arguments(cls, parser):
        return parser, set()

		
    def read_bool_status(cls, status_file):
        with open(join(cls.list_root, status_file)) as f:
            lines = f.read()
        return [x == 'True' for x in lines.split('\n')[:-1]]

    def __init__(self, opt, mode='train', model=None):
        assert mode in ('train', 'valid')
        self.mode = mode
        if model is None:
            required = ['rgb']
            self.preproc = None
        else:
            required = model.requires
            self.preproc = model.preprocess

        # Parse classes
        classes = [] 
        class_str = '' 
        for c in opt.classes.split('+'):
            class_str += c + '+'
            if c in self.class_aliases: 
                classes += self.class_aliases[c].split('+')
            else:
                classes = c.split('+')
        class_str = class_str[:-1]  
        classes = sorted(list(set(classes)))

        # split train-test
        with open(join(self.list_root, 'items.txt')) as f:
            lines = f.read()
        item_list = lines.split('\n')[:-1]
        is_train = self.read_bool_status('is_train.txt')
        assert len(item_list) == len(is_train)

		
        # Load status
        has = {}
        for data_type in required:
            assert data_type in self.status_and_suffix.keys(), \
                "%s required, but unspecified in status_and_suffix" % data_type
            has[data_type] = self.read_bool_status(
                self.status_and_suffix[data_type]['status']
            )
            assert len(has[data_type]) == len(item_list)

			
        # add all paths into a dict
        samples = []
        for i, item in enumerate(item_list):
            class_id, _ = item.split('/')[:2]
            item_in_split = ((self.mode == 'train') == is_train[i])
            if item_in_split and class_id in classes:
                # Look up subclass_id for this item
                sample_dict = {'item': join(self.data_root, item)}
                for data_type in required:
                    suffix = self.status_and_suffix[data_type]['suffix']
                    k = data_type + '_path'
                    sample_dict[k] = join(self.data_root, item + suffix) \
                            if has[data_type][i] else None
                if None not in sample_dict.values():
                    # All that are required exist
                    samples.append(sample_dict)
					
        if self.mode == 'valid':
            if opt.manual_seed:
                seed = opt.manual_seed
            else:
                seed = 0
            random.Random(seed).shuffle(samples)
        self.samples = samples

		
    def __getitem__(self, i):
        sample_loaded = {}
        for k, v in self.samples[i].items():
            sample_loaded[k] = v 
            if k.endswith('_path'):
                if v.endswith('.png'):
                    im = util.util_img.imread_wrapper(
                        v, util.util_img.IMREAD_UNCHANGED,
                        output_channel_order='RGB')
                    # Normalize to [0, 1]
                    im = im.astype(float) / float(np.iinfo(im.dtype).max)
                    sample_loaded[k[:-5]] = im
                elif v.endswith('_128.npz'):
                    sample_loaded['voxel'] = np.load(v)['voxel'][None, ...]
                else:
                    raise NotImplementedError(v)

        if self.preproc is not None:
            sample_loaded = self.preproc(sample_loaded, mode=self.mode)
        # convert all types to float32
        self.convert_to_float32(sample_loaded)
        return sample_loaded

		
    def convert_to_float32(sample_loaded):
        for k, v in sample_loaded.items():
            if isinstance(v, np.ndarray):
                if v.dtype != np.float32:
                    sample_loaded[k] = v.astype(np.float32)

    def __len__(self):
        return len(self.samples)

    def get_classes(self):
        return self._class_str
