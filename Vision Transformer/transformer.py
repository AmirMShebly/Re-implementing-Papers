import torch 
import torch.nn as nn

from patchEmbedding import PatchEmbedding
from encoder import EncoderBlock
from mlp import MLP


class ViT(nn.Module):
    def __init__(self, embeddings : PatchEmbedding, encoder: EncoderBlock, mlp: MLP, num_layers: int):
        super().__init__()
        self.embeddings_block = embeddings
        self.encoder_layer = encoder
        self.classifier = mlp
        self.num_layers = num_layers

        self.encoders = nn.ModuleList([encoder for _ in range(self.num_layers)])

    
    def forward(self, x):
        
        x = self.embeddings_block(x)
        for layer in self.encoders:
            x = layer(x)
        
        x = self.classifier(x[:, 0, :])
        
        return x 