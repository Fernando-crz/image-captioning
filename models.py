import torch
from torch import nn
from torch.nn import Module, LSTMCell, Linear, ReLU, Embedding, BatchNorm1d
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision.models import vgg16, VGG16_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors

def pack_to_multibatch(captions, captions_lengths):
    multibatch = []
    pack = pack_padded_sequence(captions, captions_lengths, batch_first=True, enforce_sorted=False)
    i = 0
    for batch_size in pack.batch_sizes:
        multibatch.append(pack.data[i:i+batch_size])
        i += batch_size
    return multibatch, pack.batch_sizes, pack.sorted_indices, pack.unsorted_indices

class Encoder(Module):
    def __init__(self):
        super().__init__()
        vgg_layers = vgg16(weights=VGG16_Weights.DEFAULT).features
        vgg_layers = list(vgg_layers.children())

        vgg_layers = vgg_layers[:-1]    # retirando última camada com max pooling

        self.vgg = nn.Sequential(*vgg_layers)
        
    def train(self, freeze_initial_layers=True):
        for parameter in self.vgg.parameters():
            parameter.requires_grad = True
        
        if freeze_initial_layers:
            for parameter in self.vgg[:4].parameters():
                parameter.requires_grad = False
        
    def forward(self, x):
        # Lembrar que toda imagem deve ser transformada para o formato (3, 224, 224)!
        image_embedding = self.vgg(x)
        image_embedding = image_embedding.permute(0, 2, 3, 1)
        image_embedding = image_embedding.view(image_embedding.size(0), -1, image_embedding.size(-1))
        return image_embedding


class Attention(Module):
    def __init__(self, hidden_size, num_conv_channels, attention_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_channels = num_conv_channels
        self.attention_size = attention_size
        
        self.img_attention = Linear(num_conv_channels, self.attention_size)
        self.hidden_attention = Linear(hidden_size, self.attention_size)
        self.attention = Linear(self.attention_size, 1)
        self.beta_linear = Linear(self.hidden_size, 1)
        self.relu = ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, img_embedding, hidden):
        """
        In:
        img_embedding: (batch_size, num_conv_channels, img_emb_size)
        hidden: (batch_size, hidden_size)
        """
        img_attention = self.img_attention(img_embedding)   # (batch_size, img_emb_size, attention_size)
        hidden_attention = self.hidden_attention(hidden)    # (batch_size, attention_size)
        attention_importance = self.attention(self.relu(img_attention + hidden_attention.unsqueeze(1))).squeeze(2) # (batch_size, img_emb_size)
        beta = self.sigmoid(self.beta_linear(hidden))  # (batch_size, 1)
        attention_weights = self.softmax(attention_importance)  # (batch_size, img_emb_size)

        context = beta * (attention_weights.unsqueeze(2) * img_embedding).sum(dim=1)      # (batch_size, num_conv_channels)

        return context, attention_weights


class Decoder(Module):
    def __init__(self, hidden_size, vocab_size, emb_size, attention_size, img_emb_size, num_channels, dropout):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.img_emb_size = img_emb_size
        self.num_channels = num_channels

        self.lstm = LSTMCell(num_channels + emb_size, hidden_size)
        self.attention = Attention(hidden_size, num_channels, attention_size)
        self.embedding = Embedding(vocab_size, emb_size)
        self.lstm_fc = Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def _init_hidden_and_cell_state(self, batch_size):
        # TODO: Revisar método de inicialização de estados hidden e cell, seguindo parametros do paper
        init_hidden = torch.zeros((batch_size, self.hidden_size)).to(device)
        init_cell = torch.zeros((batch_size, self.hidden_size)).to(device)
        return init_hidden, init_cell
    
    def forward(self, img_embedding, captions, captions_lengths):
        # img_embedding: (batch_size, img_emb_size, num_channels)
        # hidden: (batch_size, hidden_size)
        # cell: (batch_size, hidden_size)
        # captions: (batch_size, max_seq_size)
        # captions_lengths: (batch_size)

        captions_batches, batch_sizes, sorted_indices, unsorted_indices = pack_to_multibatch(captions, captions_lengths)
        max_batch_size = batch_sizes[0]
        max_seq_length = len(batch_sizes)

        hidden, cell = self._init_hidden_and_cell_state(max_batch_size)
        complete_attention_weights = torch.zeros(max_batch_size, max_seq_length, self.img_emb_size).to(device)
        complete_preds = torch.zeros(max_batch_size, max_seq_length, self.vocab_size).to(device)  

        img_embedding_sorted = img_embedding[sorted_indices]

        for i, (caption_batch, batch_size) in enumerate(zip(captions_batches, batch_sizes)):
            context, attention_weights = self.attention(img_embedding_sorted[:batch_size], hidden[:batch_size]) 

            embeded_captions = self.embedding(caption_batch)        # (batch_size, emb_size)
            hidden, cell = self.lstm(torch.cat((context, embeded_captions), dim=1), (hidden[:batch_size], cell[:batch_size]))
            preds = self.lstm_fc(self.dropout(hidden))

            complete_attention_weights[:batch_size, i, :] = attention_weights   # Erro reclamando aqui
            complete_preds[:batch_size, i, :] = preds

        complete_attention_weights = complete_attention_weights[unsorted_indices]
        complete_preds = complete_preds[unsorted_indices]

        return complete_preds, complete_attention_weights, sorted_indices