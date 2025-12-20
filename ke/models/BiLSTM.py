import torch.nn as nn
from torchcrf import CRF
from .BasicModule import BasicModule

__all__ = ['BiLSTM_CRF']

class BiLSTM_CRF(BasicModule):

    def __init__(self, processor, config):
        super(BiLSTM_CRF, self).__init__(processor, config)
        vocab_size      = len(processor.w2i)
        num_labels      = len(processor.l2i)
        embedding_dim   = config.word_dim
        hidden_size     = config.hidden_dim
        drop_out        = config.drop_out
        bidirectional   = config.bidirectional
        num_layers      = config.num_layers
    
        self.embed  = nn.Embedding(vocab_size, embedding_dim)
        self.lstm   = nn.LSTM(
            input_size      = embedding_dim,
            hidden_size     = hidden_size,
            dropout         = drop_out,
            num_layers      = num_layers,
            batch_first     = True,
            bidirectional   = bidirectional,
        )
        self.linear = nn.Linear(hidden_size * 2, num_labels)
        self.crf    = CRF(num_labels, batch_first=True)

    def _get_lstm_feature(self, input):
        out = self.embed(input)
        out, _ = self.lstm(out)
        return self.linear(out)

    def forward(self, input, mask):
        out = self._get_lstm_feature(input)
        return self.crf.decode(out, mask)

    def loss_fn(self, input, target, mask):
        y_pred = self._get_lstm_feature(input)
        return -self.crf.forward(y_pred, target, mask)