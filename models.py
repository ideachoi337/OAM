import numpy as np
import torch
import math
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import init

from mamba_ssm import Mamba2

class InferenceCache:
    def __init__(self, opt, batch_size=1):
        self.batch_size = batch_size
        self.d_model = opt['hidden_dim']
        self.n_layers = opt['n_layers']
        self.d_state = opt['d_state']
        self.d_conv = opt['d_conv']
        self.expand = opt['expand']
        self.headdim = opt['headdim']
        self.d_inner = (self.expand * self.d_model)
        self.d_ssm = self.d_inner
        self.nheads = self.d_ssm // self.headdim

        self.default_conv_state = None
        self.default_ssm_state = None
        
        self.reset()
    
    def reset(self):
        if self.default_conv_state is None:
            self.reset_zero()
            return
        self.conv_state = self.default_conv_state.clone().detach()
        self.ssm_state = self.default_ssm_state.clone().detach()
    
    def reset_zero(self):
        self.conv_state = torch.zeros(
            self.batch_size, self.d_inner + 2*self.d_state, self.d_conv, device='cuda'
        )
        self.ssm_state = torch.zeros(
            self.batch_size, self.nheads, self.headdim, self.d_state, device='cuda'
        )
    
    def set_default(self):
        self.default_conv_state = self.conv_state.clone().detach()
        self.default_ssm_state = self.ssm_state.clone().detach()
        

class ResidualMambaBlock(nn.Module):
    def __init__(self, d_model, d_state=128, d_conv=5, expand=2):
        super().__init__()
        self.mamba = Mamba2(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        return x + self.mamba(self.norm(x))
    
    def step(self, x_t, state:InferenceCache):
        x_norm = self.norm(x_t)
        y_t, state.conv_state, state.ssm_state = self.mamba.step(x_norm, state.conv_state, state.ssm_state)
        return x_t + y_t, state

class DeepMamba(nn.Module):
    def __init__(self, d_model, d_state=64, d_conv=5, expand=2, n_layers=10):
        super().__init__()
        self.blocks = nn.Sequential(*[
            ResidualMambaBlock(d_model, d_state, d_conv, expand) for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        return self.final_norm(self.blocks(x))
    
    def step(self, x_t, states):
        new_states = []
        for block, state in zip(self.blocks, states):
            x_t, new_state = block.step(x_t, state)
            new_states.append(new_state)
        return self.final_norm(x_t), new_states
 
class MambaNet(torch.nn.Module):
    def __init__(self, opt):
        super(MambaNet, self).__init__()
        self.n_feature=opt["feat_dim"] 
        self.n_class=opt["num_of_class"]
        n_embedding_dim=opt["hidden_dim"]
        n_layers=opt['n_layers']
        d_state=opt['d_state']
        d_conv=opt['d_conv']
        expand=opt['expand']
        self.anchors=opt["anchors"]
        n_anchors = len(self.anchors)
        self.anchors_stride=[]
        self.best_loss=1000000
        self.best_map=0

        # FC layers for the 2 streams
        self.feature_reduction_rgb = nn.Linear(self.n_feature//2, n_embedding_dim//2)
        self.feature_reduction_flow = nn.Linear(self.n_feature//2, n_embedding_dim//2)

        self.mamba = DeepMamba(
            d_model=n_embedding_dim, 
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            n_layers=n_layers
        )

        self.classifier = nn.Sequential(nn.Linear(n_embedding_dim, n_embedding_dim), nn.ReLU(), nn.Linear(n_embedding_dim,self.n_class*n_anchors))
        self.regressor = nn.Sequential(nn.Linear(n_embedding_dim, n_embedding_dim), nn.ReLU(), nn.Linear(n_embedding_dim,2*n_anchors))

    def forward(self, inputs, use_last=True):
        batch_size, seq_len, featSize = inputs.shape
        if not use_last: assert batch_size == 1
        
        base_x_rgb = self.feature_reduction_rgb(inputs[:, :,:self.n_feature//2])
        base_x_flow = self.feature_reduction_flow(inputs[:, :,self.n_feature//2:])
        base_x = torch.cat([base_x_rgb,base_x_flow],dim=-1) # (batch_size, seq_len, embSize)

        # Get anchor feature
        anc_feature = self.mamba(base_x) # batch_size x seq_len x embSize -> batch_size x seq_len x featsize
        if use_last: anc_feature = anc_feature[:,-1,:] # batch_size x featsize
        
        anc_cls = self.classifier(anc_feature).reshape(batch_size if use_last else seq_len, -1, self.n_class) # seq_len x n_anchor x n_class
        anc_reg = self.regressor(anc_feature).reshape(batch_size if use_last else seq_len, -1, 2) # seq_len x anchor x 2
        
        return anc_cls, anc_reg

    def step(self, x_t, states:InferenceCache):
        """
        x_t: (1, 1, feat_dim), a single frame input
        state: list of states for each DeepMamba layer
        return:
            anc_cls_t: (1, n_anchor, n_class)
            anc_reg_t: (1, n_anchor, 2)
            new_state: updated state list
        """
        assert x_t.shape[0] == 1 and x_t.shape[1] == 1

        rgb = x_t[:, :, :self.n_feature//2]   # (1, 1, D/2)
        flow = x_t[:, :, self.n_feature//2:]  # (1, 1, D/2)

        rgb_proj = self.feature_reduction_rgb(rgb)    # (1, 1, hidden_dim/2)
        flow_proj = self.feature_reduction_flow(flow) # (1, 1, hidden_dim/2)
        emb = torch.cat([rgb_proj, flow_proj], dim=-1)  # (1, 1, hidden_dim)

        anc_feature, new_state = self.mamba.step(emb, states)  # feature_t: (1, 1, hidden_dim)

        # Step 4: Classifier & Regressor
        anc_cls_t = self.classifier(anc_feature).reshape(1, -1, self.n_class)  # (1, n_anchor, n_class)
        anc_reg_t = self.regressor(anc_feature).reshape(1, -1, 2)              # (1, n_anchor, 2)

        return anc_cls_t, anc_reg_t, new_state

class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float = 0.1,
                 maxlen: int = 750):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: torch.Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

class MYNET(torch.nn.Module):
    def __init__(self, opt):
        super(MYNET, self).__init__()
        self.n_feature=opt["feat_dim"] 
        n_class=opt["num_of_class"]
        n_embedding_dim=opt["hidden_dim"]
        n_enc_layer=opt["enc_layer"]
        n_enc_head=opt["enc_head"]
        n_dec_layer=opt["dec_layer"]
        n_dec_head=opt["dec_head"]
        n_seglen=opt["segment_size"]
        self.anchors=opt["anchors"]
        self.anchors_stride=[]
        dropout=0.3
        self.best_loss=1000000
        self.best_map=0
        # FC layers for the 2 streams
        
        self.feature_reduction_rgb = nn.Linear(self.n_feature//2, n_embedding_dim//2)
        self.feature_reduction_flow = nn.Linear(self.n_feature//2, n_embedding_dim//2)
        
        self.positional_encoding = PositionalEncoding(n_embedding_dim, dropout, maxlen=400)      
        
        self.encoder = nn.TransformerEncoder(
                                            nn.TransformerEncoderLayer(d_model=n_embedding_dim, 
                                                                        nhead=n_enc_head, 
                                                                        dropout=dropout, 
                                                                        activation='gelu'), 
                                            n_enc_layer, 
                                            nn.LayerNorm(n_embedding_dim))
                                            
        self.decoder = nn.TransformerDecoder(
                                            nn.TransformerDecoderLayer(d_model=n_embedding_dim, 
                                                                        nhead=n_dec_head, 
                                                                        dropout=dropout, 
                                                                        activation='gelu'), 
                                            n_dec_layer, 
                                            nn.LayerNorm(n_embedding_dim))                            
        self.classifier = nn.Sequential(nn.Linear(n_embedding_dim,n_embedding_dim), nn.ReLU(), nn.Linear(n_embedding_dim,n_class))
        self.regressor = nn.Sequential(nn.Linear(n_embedding_dim,n_embedding_dim), nn.ReLU(), nn.Linear(n_embedding_dim,2))                               
        
        self.decoder_token = nn.Parameter(torch.zeros(len(self.anchors), 1, n_embedding_dim))
        self.relu = nn.ReLU(True)
        self.softmaxd1 = nn.Softmax(dim=-1)

    def forward(self, inputs):
        # inputs - batch x seq_len x featSize
        
        base_x_rgb = self.feature_reduction_rgb(inputs[:,:,:self.n_feature//2])
        base_x_flow = self.feature_reduction_flow(inputs[:,:,self.n_feature//2:])
        base_x = torch.cat([base_x_rgb,base_x_flow],dim=-1)
        
        base_x = base_x.permute([1,0,2])# seq_len x batch x featsize x 
        
        pe_x = self.positional_encoding(base_x)
        encoded_x = self.encoder(pe_x)    
        
        decoder_token = self.decoder_token.expand(-1, encoded_x.shape[1], -1)  
        decoded_x = self.decoder(decoder_token, encoded_x) 
        decoded_x = decoded_x.permute([1, 0, 2])
        
        anc_cls = self.classifier(decoded_x)
        anc_reg = self.regressor(decoded_x)
        
        return anc_cls, anc_reg

 
class SuppressNet(torch.nn.Module):
    def __init__(self, opt):
        super(SuppressNet, self).__init__()
        n_class=opt["num_of_class"]-1
        n_seglen=opt["sup_segment_size"]
        n_embedding_dim=2*n_seglen
        dropout=0.3
        self.best_loss=1000000
        self.best_map=0
        # FC layers for the 2 streams
        
        self.mlp1 = nn.Linear(n_seglen, n_embedding_dim)
        self.mlp2 = nn.Linear(n_embedding_dim, 1)
        self.norm = nn.InstanceNorm1d(n_class)
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, inputs):
        #inputs - batch x seq_len x class
        
        base_x = inputs.permute([0,2,1])
        base_x = self.norm(base_x)
        x = self.relu(self.mlp1(base_x))
        x = self.sigmoid(self.mlp2(x))
        x = x.squeeze(-1)
        
        return x
        

