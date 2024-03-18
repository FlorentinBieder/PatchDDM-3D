"""
This is a template for implementing your own Dataset, along with concatenating the coordinates and sampling random crops
"""
import torch
import torch.nn
import torch.utils.data
import numpy as np
import os
import os.path
import nibabel


class XXXDataset(torch.utils.data.Dataset):
    def __init__(self, random_half_crop=False, concat_coords=False):
        self.random_half_crop = random_half_crop
        self.concat_coords = concat_coords
        self.coord_cache = None
        self.image_size = 256
        """
        TODO implement your own dataset here
        """
    def __len__(self):
        return 10 # to be implemented

    def __getitem__(self, x):
        """
        TOOD implement your own image loading here
        """
        # placeholder for image we would other wise load above
        image = torch.zeros((4, self.image_size, self.image_size, self.image_size))
        label = torch.zeros((1, self.image_size, self.image_size, self.image_size))

        # create normalized coordinate system for 
        if self.concat_coords:
            if self.coord_cache is None: # cache coordinates, so we don't have to generate them from scratch each time
                dim = len(image.shape) - 1  # 2d or 3d
                self.coord_cache = torch.stack(torch.meshgrid(dim * [torch.linspace(-1, 1, self.image_size)], indexing='ij'), dim=0)
            image = torch.cat([image, self.coord_cache], dim=0)

        # half crop: crop to a 128x128[x128] image,
        if self.random_half_crop:
            shape = (len(image.shape)-1,)

            # the following represents the distribution in the paper, requires a resolution of256x256x256
            first_coords = np.random.randint(0, 32+1, shape) + np.random.randint(0, 64+32+1, shape) 
            index = tuple([slice(None), *(slice(f, f+128) for f in first_coords)])

            # the following represents an unform distribution (comment the two lines above, and uncomment the ones below)
            # first_coords = np.random.randint(0, (half_size := self.image_size//2), shape)   # alternatively an uniform distribution
            # index = tuple([slice(None), *(slice(f, f+self.image_size - half_size) for f in first_coords)])

            image = image[index]
            label = label[index]

        out_dict = dict() # for additional meta data
        out_dict["y"] = 0
        number = x
        return (image, out_dict, 0, label, number )


if __name__ == '__main__':
    ds = XXXDataset(random_half_crop=True, concat_coords=True)
    for k in range(1000):
        x = ds[3]
        x = x[0]
        print(x.shape)
