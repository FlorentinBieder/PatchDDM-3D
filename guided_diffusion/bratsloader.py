import torch
import torch.nn
import torch.utils.data
import numpy as np
import os
import os.path
import nibabel


class BRATSDataset(torch.utils.data.Dataset):
    def __init__(self, directory, test_flag=False, normalize=None, mode='train',
                 half_resolution=False, random_half_crop=False, concat_coords=False, num_classes=1):
        '''
        directory is expected to contain some folder structure:
                  if some subfolder contains only files, all of these
                  files are assumed to have a name like
                  brats_train_NNN_XXX_123_w.nii.gz
                  where XXX is one of t1, t1ce, t2, flair, seg
                  we assume these five files belong to the same image
                  seg is supposed to contain the segmentation
        half_resolution subsamples a 256^{2,3} image by a factor of 2
        random_half_crop crops a subvolume of 128^{2,3}, with a higher probability around the central region
        concat_coords = True concatenates normalized coordinates
        num_classes defines the number of classes (and therefore number of channels) of the labels, currently 1 or 4
        '''
        super().__init__()
        self.mode = mode
        self.database_dict = dict(train=[], validation=[], test=[])
        assert self.mode in self.database_dict or self.mode == 'legacy', f"invalid mode argument {self.mode}"
        if half_resolution and random_half_crop:
            raise RuntimeError(f"you probably don't want {half_resolution=} AND {random_half_crop=}")
        self.half_resolution = half_resolution
        self.random_half_crop = random_half_crop
        self.concat_coords = concat_coords
        self.num_classes = num_classes
        assert num_classes in [1, 4], f'num_classes must be 1 or 4, but it was set to {num_classes=}'
        self.coord_cache = None   # field for caching the coordinates that we append ine very step
        self.directory = os.path.expanduser(directory)
        self.normalize = normalize or (lambda x: x)
        self.test_flag=test_flag
        if test_flag:
            self.seqtypes = ['t1', 't1ce', 't2', 'flair']
        else:
            self.seqtypes = ['t1', 't1ce', 't2', 'flair', 'seg']

        self.seqtypes_set = set(self.seqtypes)
        self.database = []
        for root, dirs, files in os.walk(self.directory):
            # if there are no subdirs, we have a datadir
            if not dirs:
                files.sort()
                datapoint = dict()
                # extract all files as channels
                for f in files:
                    seqtype = f.split('_')[3]
                    datapoint[seqtype] = os.path.join(root, f)
                assert set(datapoint.keys()) == self.seqtypes_set, \
                    f'datapoint is incomplete, keys are {datapoint.keys()}'
                self.database.append(datapoint)

        # split the complete database into splits
        split = make_split()  # train, validation test
        for filedict in self.database:
            number = int(filedict['t1'].split('/')[-2])
            for mode, numbers in split.items():
                if number in numbers:
                    self.database_dict[mode].append(filedict)
                    numbers.remove(number)
                    break
            else:
                raise RuntimeError(f"number {number} not found in any of the splits")
        assert all(not numbers for numbers in split.values()), "not all numbers were distributed"
        self.database_dict['legacy'] = self.database

    def __getitem__(self, x):
        out = []
        filedict = self.database_dict[self.mode][x]
        number = filedict['t1'].split('/')[-2]
        for seqtype in self.seqtypes:
            nib_img = nibabel.load(filedict[seqtype])
            out.append(torch.tensor(nib_img.get_fdata()))
        out = torch.stack(out)
        out_dict = {}
        if self.test_flag:
            path2 = './data/brats/test_labels/' + str(number) + '-label.nii.gz'
            seg = nibabel.load(path2)
            np_seg = seg.get_fdata()
            if len(out.shape) == 3: #2d
                image = torch.zeros(4, 256, 256)
                seg = torch.zeros(1, 256, 256)
                image[:, 8:-8, 8:-8] = out		#pad to a size of (256,256)
                seg[:, 8:-8, 8:-8] = np_seg		#pad to a size of (256,256)
            elif len(out.shape) == 4: #3d
                image = out.float() #(already 256^3)
            else:
                raise ValueError(f"cannot deal with data of shape {out.shape=}")
            label = seg[None, ...]
        else:
            if len(out.shape) == 3: #2d
                image = torch.zeros(4, 256, 256)
                image[:, 8:-8, 8:-8] = out[:-1, ...]  #pad to a size of (256,256)
                seg = torch.zeros(1, 256, 256)
                seg[:, 8:-8, 8:-8] = out[-1:, ...]  #pad to a size of (256,256)
            elif len(out.shape) == 4: #3d
                image = torch.zeros(4, 256, 256, 256)
                image[:, 8:-8, 8:-8, 50:-51] = out[:-1, ...]  #pad to a size of (256,256)
                seg = torch.zeros(1, 256, 256, 256)
                seg[:, 8:-8, 8:-8, 50:-51] = out[-1:, ...]  #pad to a size of (256,256)
            else:
                raise ValueError(f"cannot deal with data of shape {out.shape=}")
            label = seg

        # normalization
        image = self.normalize(image)

        weak_label = int(label.max() > 0)

        # normalized coordinates
        if self.concat_coords:
            if self.coord_cache is None:
                dim = len(image.shape) - 1  # 2d or 3d
                self.coord_cache = torch.stack(torch.meshgrid(dim * [torch.linspace(-1, 1, 256)], indexing='ij'), dim=0)
            image = torch.cat([image, self.coord_cache], dim=0)

        # half resolution, temporary
        if self.half_resolution:
            if len(image.shape) == 4:
                image = image[:, ::2, ::2, ::2]
                label = label[:, ::2, ::2, ::2]
            elif len(image.shape) == 3:
                image = image[:, ::2, ::2]
                label = label[:, ::2, ::2]
            else:
                raise ValueError(f"cannot deal with data of shape {out.shape=}")
        # half crop: crop to a 128x128[x128] image,
        if self.random_half_crop:
            shape = (len(image.shape)-1,)
            first_coords = np.random.randint(0, 32+1, shape) + np.random.randint(0, 64+32+1, shape)
            index = tuple([slice(None), *(slice(f, f+128) for f in first_coords)])
            image = image[index]
            label = label[index]

        if self.num_classes == 1:  # merge all in to one class
            label = (label > 0).float()
        elif self.num_classes == 4:  # use separate channels for each class (0=bg)
            new_label = label == torch.tensor([0, 1, 2, 4])[(slice(None), *(len(label.shape)-1)*(None,))]
            label = new_label.float()
        else:
            raise ValueError(f'{self.num_classes=} should be in 1 or 4')

        out_dict["y"] = weak_label
        return (image, out_dict, weak_label, label, number )

    def __len__(self):
        return len(self.database_dict[self.mode])


"""
split train / validation / test as 80% / 10% / 10%
for 369 scans this is 295 / 37 / 37
"""
def make_split(split = [295, 37, 37], total=369, seed=0):
    assert sum(split) == total, "incorrect number in splits"
    import numpy as np
    rng = np.random.RandomState(seed)
    a = np.arange(1, total+1)
    rng.shuffle(a)
    train, validation, test = a[:split[0]], a[split[0]:split[0]+split[1]], a[split[0]+split[1]:]
    train.sort(), validation.sort(), test.sort()
    assert(set(a) == set(train) | set(test) | set(validation)), "sets not partitioned!!!!"
    return dict(train=train.tolist(), validation=validation.tolist(), test=test.tolist())

if __name__ == '__main__':
    # testing
    split = make_split()
    print(split['validation'])
    ds = BRATSDataset(
        directory='~/datasets/BRATS2020_nifti3d/train',
        test_flag=False,
        normalize=None,
    )
    for i in range(3):
        (image, out_dict, weak_label, label, number) = ds[i]
        print(image.shape, label.shape)





