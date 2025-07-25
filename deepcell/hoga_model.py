import torch
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, LayerNorm, Dropout, Softmax
from collections import defaultdict
from .arch.mlp import MLP

'''
Slightly modified multihead attention for Gamora
'''
class MultiheadAttentionMix(torch.nn.Module):
    def __init__(self, input_dim, num_heads, dropout=0.0):
        super(MultiheadAttentionMix, self).__init__()

        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        # Linear projections for queries, keys, and values
        self.query_projection = Linear(input_dim, input_dim)
        self.key_projection = Linear(input_dim, input_dim)
        self.value_projection = Linear(input_dim, input_dim)

        # Linear projection for the output of the attention heads
        self.output_projection = Linear(input_dim, input_dim)

        self.dropout = Dropout(dropout)
        self.softmax = Softmax(dim=-1)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear projections for queries, keys, and values
        query = self.query_projection(query)
        key = self.key_projection(key)
        value = self.value_projection(value)

        # Reshape the projected queries, keys, and values
        query = query.view(batch_size * self.num_heads, -1, self.head_dim)
        key = key.view(batch_size * self.num_heads, -1, self.head_dim)
        value = value.view(batch_size * self.num_heads, -1, self.head_dim)

        # Compute the scaled dot-product attention
        attention_scores = torch.bmm(query, key.transpose(1, 2))
        attention_scores = attention_scores / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))

        # Apply the mask (if provided)
        if mask is not None:
            mask = mask.unsqueeze(1)  # Add head dimension
            attention_scores = attention_scores.masked_fill(mask == 0, float("-inf"))

        attention_probs = self.softmax(attention_scores)
        attention_probs = self.dropout(attention_probs)

        # Compute the output of the attention heads
        attention_output = torch.bmm(attention_probs, value)

        # Reshape and project the output of the attention heads
        attention_output = attention_output.view(batch_size, -1, self.input_dim)
        attention_output = self.output_projection(attention_output)

        return attention_output, attention_probs

'''
Vanilla multihead attention (recommended for general use cases)
'''
class MultiheadAttention(torch.nn.Module):
    def __init__(self, input_dim, num_heads, dropout=0.0):
        super(MultiheadAttention, self).__init__()

        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        # Linear projections for queries, keys, and values
        self.query_projection = Linear(input_dim, input_dim)
        self.key_projection = Linear(input_dim, input_dim)
        self.value_projection = Linear(input_dim, input_dim)

        # Linear projection for the output of the attention heads
        self.output_projection = Linear(input_dim, input_dim)

        self.dropout = Dropout(dropout)
        self.softmax = Softmax(dim=-1)

    def forward(self, query, key, value, mask=None):
        batch_size, seq_len, _ = query.size()

        # Linear projections for queries, keys, and values
        query = self.query_projection(query)
        key = self.key_projection(key)
        value = self.value_projection(value)

        # Reshape the projected queries, keys, and values
        query = query.view(batch_size, seq_len, self.head_dim, -1)
        key = key.view(batch_size, seq_len, self.head_dim, -1)
        value = value.view(batch_size, seq_len, self.head_dim, -1)

        # Compute the scaled dot-product attention
        attention_scores = torch.einsum('bldh, bndh -> blnh', query, key)
        attention_scores = attention_scores / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))

        attention_probs = self.softmax(attention_scores)
        attention_probs = self.dropout(attention_probs)

        # Compute the output of the attention heads
        attention_output = torch.einsum('blnh, bndh -> bldh', attention_probs, value)

        # Reshape and project the output of the attention heads
        attention_output = attention_output.reshape(batch_size, seq_len, self.input_dim)
        attention_output = self.output_projection(attention_output)

        return attention_output, attention_probs

class HOGA(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, num_hops, heads, directed, attn_dropout=0.0, attn_type="vanilla", use_bias=False):
        super(HOGA, self).__init__()
        self.num_layers = num_layers
        self.num_hops = num_hops
        self.directed = directed

        self.lins = torch.nn.ModuleList()
        self.gates = torch.nn.ModuleList()
        self.trans = torch.nn.ModuleList()
        self.lns = torch.nn.ModuleList()
        self.lins.append(Linear(in_channels, hidden_channels, bias=use_bias))
        self.lins.append(Linear(hidden_channels, hidden_channels, bias=use_bias))
        self.lins.append(Linear(hidden_channels, hidden_channels, bias=use_bias))
        self.gates.append(Linear(hidden_channels, hidden_channels, bias=use_bias))
        if attn_type == "vanilla":
            self.trans.append(MultiheadAttention(hidden_channels, heads, dropout=attn_dropout))
        else:
            self.trans.append(MultiheadAttentionMix(hidden_channels, heads, dropout=attn_dropout))
        self.lns.append(LayerNorm(hidden_channels))
        for _ in range(num_layers - 1):
            self.lins.append(Linear(hidden_channels, hidden_channels, bias=use_bias))
            self.gates.append(Linear(hidden_channels, hidden_channels, bias=use_bias))
            if attn_type == "vanilla":
                self.trans.append(MultiheadAttention(hidden_channels, heads, dropout=attn_dropout))
            else:
                self.trans.append(MultiheadAttentionMix(hidden_channels, heads, dropout=attn_dropout))
            self.lns.append(LayerNorm(hidden_channels))

        # Linear layers for predictions
        self.linear = torch.nn.ModuleList()
        self.linear.append(Linear(hidden_channels, hidden_channels, bias=use_bias))
        self.linear.append(Linear(hidden_channels, out_channels, bias=use_bias))
        # self.linear.append(Linear(hidden_channels, out_channels, bias=use_bias))
        # self.linear.append(Linear(hidden_channels, out_channels, bias=use_bias))

        self.bn = BatchNorm1d(hidden_channels)
        self.attn_layer = Linear(2 * hidden_channels, 1)
        
        self.lin_hs_hf = Linear(hidden_channels, hidden_channels * 2, bias=use_bias)
        self.readout_prob = MLP(hidden_channels, 32, 1, num_layer=3, p_drop=0.2, norm_layer='batchnorm',
                                act_layer='relu')
        
        
        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for gate in self.gates:
            gate.reset_parameters()
        for li in self.linear:
            li.reset_parameters()
        self.bn.reset_parameters()

    def forward(self, g):
        # Current implementation: use a shared linear layer for all hop-wise features
        # Note: apply separate layers for different hop-wise features may further improve accuracy
        x = g.aig_hop_x
        
        x = self.lins[0](x)

        for i, tran in enumerate(self.trans):
            x = self.lns[i](self.gates[i](x)*(tran(x, x, x)[0]))
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        if not self.directed:
            target = x[:,0,:].unsqueeze(1).repeat(1,(self.num_hops-1),1)
            split_tensor = torch.split(x, [1, (self.num_hops-1)], dim=1)
        else:
            target = x[:,0,:].unsqueeze(1).repeat(1,(self.num_hops-1)*2,1)
            split_tensor = torch.split(x, [1, (self.num_hops-1)*2], dim=1)
        node_tensor = split_tensor[0]
        neighbor_tensor = split_tensor[1]
        layer_atten = self.attn_layer(torch.cat((target, neighbor_tensor), dim=2))
        layer_atten = F.softmax(layer_atten, dim=1)
        neighbor_tensor = neighbor_tensor * layer_atten
        neighbor_tensor = torch.sum(neighbor_tensor, dim=1, keepdim=True)
        x = (node_tensor + neighbor_tensor).squeeze()
        x = self.linear[0](x)
        x = self.bn(F.relu(x))
        x = F.dropout(x, p=self.dropout, training=self.training)

        x1 = self.linear[1](x) # for xor
        # x2 = self.linear[2](x) # for maj
        # x3 = self.linear[3](x) # for roots
        
        state = self.lin_hs_hf(x)
        hs = state[:, :state.size(1)//2]
        hf = state[:, state.size(1)//2:]
        
        return hs, hf
    
    def pred_prob(self, hf):
        prob = self.readout_prob(hf)
        prob = torch.clamp(prob, min=0.0, max=1.0)
        return prob
    
    def load(self, model_path):
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
        state_dict_ = checkpoint['state_dict']
        state_dict = {}
        for k in state_dict_:
            if k.startswith('module') and not k.startswith('module_list'):
                state_dict[k[7:]] = state_dict_[k]
            else:
                state_dict[k] = state_dict_[k]
        model_state_dict = self.state_dict()
        
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    print('Skip loading parameter {}, required shape{}, loaded shape{}.'.format(
                        k, model_state_dict[k].shape, state_dict[k].shape))
                    state_dict[k] = model_state_dict[k]
            else:
                print('Drop parameter {}.'.format(k))
        for k in model_state_dict:
            if not (k in state_dict):
                print('No param {}.'.format(k))
                state_dict[k] = model_state_dict[k]
        self.load_state_dict(state_dict, strict=False)
      