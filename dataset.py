import os
import json
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Resize, CenterCrop, Compose, ToTensor
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
from PIL import Image

class CocoDS(Dataset):
    def __init__(self, imgs_path, annotations_path, vocab=None):
        annotations_json = json.loads(open(annotations_path, "r").read())
        img_id_to_path = {img["id"]: img["file_name"] for img in annotations_json["images"]}

        data_raw = []
        for anno in annotations_json["annotations"]:
            file_name = img_id_to_path[anno["image_id"]]
            data_raw.append((os.path.join(imgs_path, file_name), anno["caption"]))

        tokenizer = get_tokenizer("basic_english")

        data_tokenized = [(img_path, ["<bos>"] + tokenizer(caption) + ["<eos>"]) for img_path, caption in data_raw]

        if vocab is None:
            self.vocab = build_vocab_from_iterator(
                [caption for _, caption in data_tokenized],
                min_freq=5,
                specials=["<pad>", "<unk>", "<bos>", "<eos>"]
                )
            self.vocab.set_default_index(self.vocab["<unk>"])
        else:
            self.vocab = vocab

        data_lens = [(img_path, caption, len(caption)) for img_path, caption in data_tokenized]

        max_len = max(data_lens, key=lambda x: x[2])[2]
        
        self.data = [(img_path, self.vocab(caption + ["<pad>"] * (max_len - caption_len)), caption_len) 
                     for img_path, caption, caption_len in data_lens]

        self.img_transform = Compose([
                            ToTensor(),
                            Resize(256, antialias=True),
                            CenterCrop(224)
                            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = Image.open(self.data[index][0])
        img = self.img_transform(img)
        if img.size(0) == 1:
            img = img.expand(3, *img.size()[1:])
        
        return img, torch.tensor(self.data[index][1]), self.data[index][2]


class CocoTestDS(Dataset):
    # classe muito similar a CocoDS, mas utilizada para validação e teste do modelo. a mesma já retorna seus dados em batch,
    # porém cada batch corresponde à apenas uma única imagem com várias captações.
    def __init__(self, imgs_path, annotations_path, vocab=None):
        annotations_json = json.loads(open(annotations_path, "r").read())
        img_id_to_path = {img["id"]: img["file_name"] for img in annotations_json["images"]}

        data_raw = []
        for anno in annotations_json["annotations"]:
            file_name = img_id_to_path[anno["image_id"]]
            data_raw.append((os.path.join(imgs_path, file_name), anno["caption"]))

        tokenizer = get_tokenizer("basic_english")

        data_tokenized = [(img_path, ["<bos>"] + tokenizer(caption) + ["<eos>"]) for img_path, caption in data_raw]

        if vocab is None:
            self.vocab = build_vocab_from_iterator(
                [caption for _, caption in data_tokenized],
                min_freq=5,
                specials=["<pad>", "<unk>", "<bos>", "<eos>"]
                )
            self.vocab.set_default_index(self.vocab["<unk>"])
        else:
            self.vocab = vocab

        data_lens = [(img_path, caption, len(caption)) for img_path, caption in data_tokenized]

        max_len = max(data_lens, key=lambda x: x[2])[2]
        
        data = [(img_path, self.vocab(caption + ["<pad>"] * (max_len - caption_len)), caption_len) 
                     for img_path, caption, caption_len in data_lens]
        
        self.batched_data = {}
        for img_path, encoded_seq, cap_len in data:
            if img_path not in self.batched_data:
                self.batched_data[img_path] = []
            
            self.batched_data[img_path].append((img_path, torch.tensor(encoded_seq), cap_len))

        self.batched_data = list(self.batched_data.values())

        self.img_transform = Compose([
                            ToTensor(),
                            Resize(256, antialias=True),
                            CenterCrop(224)
                            ])
    
    def __len__(self):
        return len(self.batched_data)

    def __getitem__(self, index):
        data = self.batched_data[index]
        batch_size = len(data)
        img = Image.open(data[0][0])
        img = self.img_transform(img)
        if img.size(0) == 1:
            img = img.expand(3, *img.size()[1:])
        img_batch = torch.zeros(batch_size, *img.size(), dtype=torch.float)
        caption_batch = torch.zeros(batch_size, *data[0][1].size(), dtype=torch.long)
        cap_len_batch = torch.zeros(batch_size, dtype=torch.int)
        for i in range(batch_size):
            img_batch[i] = img
            caption_batch[i] = data[i][1]
            cap_len_batch[i] = data[i][2]
        return img_batch, caption_batch, cap_len_batch