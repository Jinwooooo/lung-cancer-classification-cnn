# EXTERNAL IMPORT(S)
import torch, os, glob, numpy as np
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset
from torchvision.transforms import transforms

# GLOBAL STATIC VARIABLES
CATEGORY = ['NORM', 'ADC', 'SCLC', 'SC']
IMAGE_SIZE = 2560
PATCH_SIZE = 512

# Extracting Patches Class
class ExtractPatches:
    def __init__(self, image, patchSize, stride):
        self.image = image
        self.patchSize = patchSize
        self.stride = stride

    def extract_single_patches(self, patch):
        croppedPatches = self.image.crop((patch[0] * self.stride, patch[1] * self.stride,
                                          patch[0] * self.stride + self.patchSize,
                                          patch[1] * self.stride + self.patchSize))

        return croppedPatches
    def no_of_patches(self):
        xNoOfPatches, yNoOfPatches = (int((self.image.width - self.patchSize) / self.stride + 1),
                                      int((self.image.height - self.patchSize) / self.stride + 1))

        return xNoOfPatches, yNoOfPatches

    def extract_all_patches(self):
        xNoOfPatches, yNoOfPatches = self.no_of_patches()

        allPatches = list()
        for y in range(yNoOfPatches):
            for x in range(xNoOfPatches):
                allPatches.append(self.extract_single_patches((x,y)))

        return allPatches

# Patch Data Class
class PatchLevelData(Dataset):
    def __init__(self, path, stride = PATCH_SIZE, flip = False, rotate = False, enhance = False):
        super().__init__()

        xNoOfPatches, yNoOfPatches = (int((IMAGE_SIZE - PATCH_SIZE) / stride + 1), int((IMAGE_SIZE - PATCH_SIZE) / stride + 1))

        patchDataDict = {}
        for categoryIndex in range(len(CATEGORY)):
            for dir in glob.glob(path + '/' + CATEGORY[categoryIndex] + '/*.png'):
                patchDataDict[dir] = categoryIndex

        self.path = path
        self.stride = stride
        self.labels = patchDataDict
        self.names = list(sorted(patchDataDict.keys()))
        self.shape = (len(patchDataDict), xNoOfPatches, yNoOfPatches, (2 if flip else 1), (4 if rotate else 1), (2 if enhance else 1))
        self.augmentSize = np.prod(self.shape) / len(patchDataDict)

    def __getitem__(self, index):
        img, xPatches, yPatches, flip, rotation, enhance = np.unravel_index(index, self.shape)

        with Image.open(self.names[img]) as image:
            extractor = ExtractPatches(image=image, patchSize=PATCH_SIZE, stride=self.stride)
            patch = extractor.extract_single_patches((xPatches, yPatches))

            if flip != 0:
                patch = patch.transpose(Image.FLIP_LEFT_RIGHT)
            if rotation != 0:
                patch = patch.rotate(rotation * 90)
            if enhance != 0:
                factors = np.random.uniform(1, 2, 3)
                patch = ImageEnhance.Color(patch).enhance(factors[0])
                patch = ImageEnhance.Contrast(patch).enhance(factors[1])
                patch = ImageEnhance.Brightness(patch).enhance(factors[2])

            label = self.labels[self.names[img]]
            return (transforms.ToTensor()(patch), label)

    def __len__(self):
        return np.prod(self.shape)

# Image Data Class
class ImageLevelData(Dataset):
    def __init__(self, path, stride=PATCH_SIZE, flip=False, rotate=False, enhance=False):
        super().__init__()

        imageDataDict = {}
        for categoryIndex in range(len(CATEGORY)):
            for dir in glob.glob(path + '/' + CATEGORY[categoryIndex] + '/*.png'):
                imageDataDict[dir] = categoryIndex

        self.path = path
        self.stride = stride
        self.labels = imageDataDict
        self.names = list(sorted(imageDataDict.keys()))
        self.shape = (len(imageDataDict), (2 if flip else 1), (4 if rotate else 1), (2 if enhance else 1))
        self.augmentSize = np.prod(self.shape) / len(imageDataDict)

    def __getitem__(self, index):
        img, flip, rotation, enhance = np.unravel_index(index, self.shape)

        with Image.open(self.names[img]) as image:
            if flip != 0:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            if rotation != 0:
                image = image.rotate(rotation * 90)
            if enhance != 0:
                factors = np.random.uniform(1, 2, 3)
                image = ImageEnhance.Color(image).enhance(factors[0])
                image = ImageEnhance.Contrast(image).enhance(factors[1])
                image = ImageEnhance.Brightness(image).enhance(factors[2])

            extractor = ExtractPatches(image=image, patchSize=PATCH_SIZE, stride=self.stride)
            patches = extractor.extract_all_patches()

            label = self.labels[self.names[img]]

            b = torch.zeros((len(patches), 3, PATCH_SIZE, PATCH_SIZE))
            for i in range(len(patches)):
                b[i] = transforms.ToTensor()(patches[i])

            return (b, label)

    def __len__(self):
        return np.prod(self.shape)

# Test Data Class
class TestData(Dataset):
    def __init__(self, path, stride=PATCH_SIZE, augment=False):
        super().__init__()

        if os.path.isdir(path):
            names = [name for name in glob.glob(path + '/*.png')]
        else:
            names = [path]

        self.path = path
        self.stride = stride
        self.augment = augment
        self.names = list(sorted(names))

    def __getitem__(self, index):
        file = self.names[index]
        with Image.open(file) as image:
            bins = 8
            if self.augment == True:
                bins = 8
            else:
                bins = 1
            extractor = ExtractPatches(image=image, patchSize=PATCH_SIZE, stride=self.stride)
            b = torch.zeros((bins, extractor.no_of_patches()[0] * extractor.no_of_patches()[1], 3, PATCH_SIZE, PATCH_SIZE))

            for k in range(bins):
                if k % 4 != 0:
                    image = image.rotate((k % 4) * 90)
                if k // 4 != 0:
                    image = image.transpose(Image.FLIP_LEFT_RIGHT)

                extractor = ExtractPatches(image=image, patchSize=PATCH_SIZE, stride=self.stride)
                patches = extractor.extract_all_patches()

                for i in range(len(patches)):
                    b[k,i] = transforms.ToTensor()(patches[i])

            return (b, file)

    def __len__(self):
        return len(self.names)
