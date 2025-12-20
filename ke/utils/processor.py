import torch
from .ioUtils import read_file

__all__ = [
    'NERProcessor',
]

class InputExample(object):
    def __init__(self, guid, text, label=None):
        self.guid = guid
        self.text = text
        self.label = label

class NERProcessor(object):
    def __init__(self, config):
        self.train_examples = self._create_examples(read_file(config.data.src.train), 'train')
        self.valid_examples = self._create_examples(read_file(config.data.src.valid ), 'valid')
        self.w2i = {}   # word  to index
        self.l2i = {}   # label to index
        self.i2l = {}   # index to label
        for example in self.train_examples:
            text_list  = example.text.split(' ')
            label_list = example.label
            for text in text_list:
                if text not in self.w2i:
                    self.w2i[text] = len(self.w2i)
            for label in label_list:
                if label not in self.l2i:
                    index = len(self.l2i)
                    self.l2i[label] = index
                    self.i2l[index] = label
        self.w2i['<unk>'] = len(self.w2i)
        self.w2i['<pad>'] = len(self.w2i)


    def _create_examples(self, data, type):
        return [
            InputExample(
                guid  = f'{type}-{i}',
                text  = ' '.join(sentence),
                label = label
            ) for i, (sentence, label) in enumerate(data)
        ]
    
    def collate(self, batch):
        max_len = max([len(item.text.split(' ')) for item in batch])
        inputs  = []
        targets = []
        masks   = []

        UNK = self.w2i.get('<unk>')
        PAD = self.w2i.get('<pad>')

        for item in batch:
            input  = [self.w2i.get(word, UNK) for word  in item.text.split(' ')]
            target = [self.l2i.get(label)     for label in item.label.copy()   ]
            assert len(input) == len(target)
            pad_len = max_len - len(input)
            inputs.append(input + [PAD] * pad_len)
            targets.append(target + [0] * pad_len)
            masks.append([True] * len(input) + [False] * pad_len)
        return torch.tensor(inputs), torch.tensor(targets), torch.tensor(masks)
