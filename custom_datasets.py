import json
import os
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from pytorch_lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader, random_split
from torch.utils.data._utils.collate import default_collate
from torchvision import transforms
from torchvision.datasets import VisionDataset

from captioning.utils import filter_out_test

TargetType = Dict[str, Union[Tensor, str, int]]


class MultiLabelImageDataset(VisionDataset):
    """Dataset that indexes images and keywords.

    :param root: path of folder containing images
    :type root: str
    :param (attr/class/desc)_data: dataframe where each row is a product
    :type (attr/class/desc)_data: pd.DataFrame
    """

    def __init__(
        self,
        tokenizer,
        root: Union[str, Path],
        desc_data: pd.DataFrame,
        class_data: pd.DataFrame,
        attr_data: pd.DataFrame,
        transform: Callable[[Image.Image], Tensor] = None,
        target_transform: Callable[[TargetType], TargetType] = None,
        transforms: Callable[
            [Image.Image, TargetType], Tuple[Tensor, TargetType]
        ] = None,
        only_frontal: bool = True,
        eostoken="<|endoftext|>",
    ):

        super().__init__(
            root=root,
            transform=transform,
            transforms=transforms,
            target_transform=target_transform,
        )
        self.tokenizer = tokenizer
        self.eostoken = eostoken
        self.img_fp = root
        # removing missing images ID from CSVs and creating multi-image index
        files = os.listdir(self.img_fp)
        if only_frontal:
            files = [image for image in files if image[-6:] == "-F.jpg"]
        dedup_files = {filename[:-6] for filename in files}
        desc_data = desc_data[desc_data["productId"].isin(dedup_files)]
        self.classes_names = np.asarray(
            [col for col in class_data.columns if col != "productId"]
        )
        self.classes_names = np.array([x.split("_")[-1] for x in self.classes_names])

        self.attr_names = np.asarray(
            [col for col in attr_data.columns if col != "productId"]
        )
        self.class_data = torch.tensor(
            class_data[class_data["productId"].isin(dedup_files)]
            .drop(["productId"], axis=1)
            .to_numpy()
        )
        self.attr_data = torch.tensor(
            attr_data[attr_data["productId"].isin(dedup_files)]
            .drop(["productId"], axis=1)
            .to_numpy()
        )
        dedup_files &= set(desc_data["productId"].to_list())
        self.index2img = [file for file in files if file[:-6] in dedup_files]

        # TODO: turning words into indices (maybe not need for gpt2 tokenizer)
        desc_data = desc_data.set_index("productId").to_dict("index")
        # vocab: Set[str] = {label for k, labels in desc_data.items() for label in labels}
        # self.vocab = {word: idx for idx, word in enumerate(vocab)}
        # indexed_data = {
        #    k: [self.vocab[label] for label in labels] for k, labels in desc_data.items()
        # }
        self.pid2pos = {k: pos for pos, (k, v) in enumerate(desc_data.items())}
        self.data = desc_data

        # self.data = torch.tensor([vec for vec in indexed_data])

        assert len(self.data) == len(self.class_data)
        assert len(self.data) == len(self.attr_data)

    def __len__(self) -> int:
        """Dataset length.

        :return: Number of items in the dataset
        :rtype: int
        """
        return len(self.index2img)

    @property
    def n_products(self) -> int:
        """Return the number of products."""
        return len(self.pid2pos)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Dict[str, Union[Tensor, str]]]:
        """Return an item given its position in the dataset.

        :param idx: Index
        :type idx: int
        :return: (image, target). Target is a list of captions for the image.
        :rtype: Tuple[Tensor, Tensor]
        """
        img_key = str(self.index2img[idx])
        pid = img_key[:-6]
        img = Image.open(os.path.join(self.img_fp, img_key)).convert("RGB")

        classes_true = self.class_data[self.pid2pos[pid]]
        attrs_true = self.attr_data[self.pid2pos[pid]]
        desc = self.data[pid]["long_description_ec_en"]

        # kw = self.get_keywords(classes_true, attrs_true)

        # keywords = self.tokenizer(kw, padding=False, return_tensors="pt")["input_ids"]

        text_labels = self.tokenizer(desc, padding=False, return_tensors="pt")[
            "input_ids"
        ]
        pad_id = (
            torch.tensor(
                self.tokenizer.convert_tokens_to_ids("<|endoftext|>"),
                dtype=text_labels.dtype,
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )
        text_labels = torch.cat((pad_id, text_labels), dim=-1)
        target = torch.cat((text_labels, pad_id), dim=-1)
        target = target.squeeze(0)

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        elif self.transform is not None:
            img = self.transform(img)
        elif self.target_transform is not None:
            target = self.target_transform(target)
        return img, target  # type: ignore[no-any-return]

    def get_keywords(
        self, classes_true: Tensor, attributes_true: Tensor,
    ):

        c_names = np.where(classes_true.cpu() > 0, self.classes_names, "<|endoftext|>",)
        c_labels = " ".join(w for w in c_names if w != "<|endoftext|>")

        a_names = np.where(attributes_true.cpu() > 0, self.attr_names, "<|endoftext|>",)
        a_labels = " ".join(str(w) for w in a_names if w != "<|endoftext|>")
        kw = c_labels + " " + a_labels

        # if self.rm_stop_words:
        #     kw = [remove_stopwords(k) for k in kw]
        return kw


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.
    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    lengths = torch.LongTensor(lengths).unsqueeze(1)
    print(lengths.shape)
    print(targets.shape)
    return images, targets, lengths


class BrunelloImageDataModule(LightningDataModule):
    """DataModule that sets loaders for tagging and captioning."""

    def __init__(
        self,
        tokenizer,
        base_path: str = "data/brunello",
        train_split_size: float = 0.9,
        batch_size: int = 4,
        num_workers: int = 4,
        collate: Callable[
            [List[Tuple[Tensor, TargetType]]], Tuple[Tensor, TargetType]
        ] = None,
        nrows=None,
        **dataset_kwargs,
    ):
        super().__init__()
        self.base_path = base_path
        self.train_split_size = train_split_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collate = collate
        self.tokenizer = tokenizer

    def setup(self, stage=None):
        """Set up datasets."""
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        train_transformations = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
        test_transformations = transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor(), normalize]
        )
        data_path = Path(self.base_path)
        desc = pd.read_csv(os.path.join(data_path, "target_desc.csv"))
        to_keep = filter_out_test(desc)
        desc = desc.iloc[to_keep, :]
        dummy = pd.read_csv(os.path.join(data_path, "dummy.csv")).iloc[to_keep, :]
        cols = [col for col in dummy.columns if "model_" not in col]

        dummy = dummy[cols]
        dummy_bullet = pd.read_csv(os.path.join(data_path, "dummy_bullet.csv")).iloc[
            to_keep, :
        ]

        dataset = MultiLabelImageDataset(
            tokenizer=self.tokenizer,
            root=data_path.joinpath("images/train"),
            class_data=dummy,
            attr_data=dummy_bullet,
            desc_data=desc,
            transform=train_transformations,
        )
        self.attrs_names = dataset.attr_names
        self.classes_names = np.array([x.split("_")[-1] for x in dataset.classes_names])

        train_len = int(self.train_split_size * len(dataset))

        train_dataset, valid_dataset = random_split(
            dataset, (train_len, len(dataset) - train_len)
        )

        valid_dataset.dataset.transform = test_transformations

        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset

        self.test_dataset = MultiLabelImageDataset(
            tokenizer=self.tokenizer,
            root=data_path.joinpath("images/test"),
            class_data=dummy,
            attr_data=dummy_bullet,
            desc_data=desc,
            transform=test_transformations,
        )

    def __dataloader(self, train):
        """Train/validation loaders."""
        _dataset = self.train_dataset if train else self.valid_dataset
        loader = DataLoader(
            dataset=_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True if train else False,
            drop_last=True,
            collate_fn=self.collate,
        )

        return loader

    def train_dataloader(self):  # noqa D102

        return self.__dataloader(train=True)

    def val_dataloader(self):  # noqa D102

        return self.__dataloader(train=False)

    def test_dataloader(self):
        """Return dataloader wrapping test dataset."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
            collate_fn=self.collate,
        )

    def get_class_names(self):
        return self.classes_names

    def get_attr_names(self):
        return self.attrs_names
