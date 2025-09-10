import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

class Fusion(nn.Module):
    """docstring for Fusion"""
    def __init__(self, d_hid):
        super(Fusion, self).__init__()
        self.r = nn.Linear(d_hid*3, d_hid, bias=False)
        self.g = nn.Linear(d_hid*3, d_hid, bias=False)

    def forward(self, x, y):
        r_ = self.r(torch.cat([x,y,x-y], dim=-1))#.tanh()
        g_ = torch.sigmoid(self.g(torch.cat([x,y,x-y], dim=-1)))
        return g_ * r_ + (1 - g_) * x

class QueryReform(nn.Module):
    """docstring for QueryReform"""
    def __init__(self, h_dim):
        super(QueryReform, self).__init__()
        # self.q_encoder = AttnEncoder(h_dim)
        self.fusion = Fusion(h_dim)
        self.q_ent_attn = nn.Linear(h_dim, h_dim)

    def forward(self, q_node, ent_emb, seed_info, ent_mask):
        '''
        q: (B,q_len,h_dim)
        q_mask: (B,q_len)
        q_ent_span: (B,q_len)
        ent_emb: (B,C,h_dim)
        seed_info: (B, C)
        ent_mask: (B, C)
        '''
        # q_node = self.q_encoder(q, q_mask)
        q_ent_attn = (self.q_ent_attn(q_node).unsqueeze(1) * ent_emb).sum(2, keepdim=True)
        q_ent_attn = F.softmax(q_ent_attn - (1 - ent_mask.unsqueeze(2)) * 1e8, dim=1)
        attn_retrieve = (q_ent_attn * ent_emb).sum(1)

        seed_retrieve = torch.bmm(seed_info.unsqueeze(1), ent_emb).squeeze(1) # (B, 1, h_dim)
        # how to calculate the gate

        #return  self.fusion(q_node, attn_retrieve)
        return  self.fusion(q_node, seed_retrieve)

class AttnEncoder(nn.Module):
    def __init__(self, d_hid):
        super(AttnEncoder, self).__init__()
        self.attn_linear = nn.Linear(d_hid, 1, bias=False)

    def forward(self, x, x_mask):
        """
        x: (B, len, d_hid)
        x_mask: (B, len)
        return: (B, d_hid)
        """
        x_attn = self.attn_linear(x)  # (B, len, 1)

        # safe masking
        x_attn = x_attn.masked_fill(x_mask.unsqueeze(2) == 0, -1e9)

        # 防止 all mask=0 → softmax NaN
        all_mask_zero = (x_mask.sum(dim=1) == 0)  # (B,)
        if all_mask_zero.any():
            # 找到這些 batch，直接給 uniform 分布
            for i in torch.where(all_mask_zero)[0]:
                x_attn[i] = 0.0

        x_attn = F.softmax(x_attn, dim=1)
        x_attn = torch.nan_to_num(x_attn, nan=0.0, posinf=0.0, neginf=0.0)

        out = (x * x_attn).sum(1)
        out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)

        return out


class Attention(nn.Module):
    """ Applies attention mechanism on the `context` using the `query`.

    **Thank you** to IBM for their initial implementation of :class:`Attention`. Here is
    their `License
    <https://github.com/IBM/pytorch-seq2seq/blob/master/LICENSE>`__.

    Args:
        dimensions (int): Dimensionality of the query and context.
        attention_type (str, optional): How to compute the attention score:

            * dot: :math:`score(H_j,q) = H_j^T q`
            * general: :math:`score(H_j, q) = H_j^T W_a q`

    Example:

         >>> attention = Attention(256)
         >>> query = torch.randn(5, 1, 256)
         >>> context = torch.randn(5, 5, 256)
         >>> output, weights = attention(query, context)
         >>> output.size()
         torch.Size([5, 1, 256])
         >>> weights.size()
         torch.Size([5, 1, 5])
    """

    def __init__(self, dimensions, attention_type='general'):
        super(Attention, self).__init__()

        if attention_type not in ['dot', 'general']:
            raise ValueError('Invalid attention type selected.')

        self.attention_type = attention_type
        if self.attention_type == 'general':
            self.linear_in = nn.Linear(dimensions, dimensions, bias=False)

        self.linear_out = nn.Linear(dimensions * 2, dimensions, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

    def forward(self, x, x_mask):
        x_attn = self.attn_linear(x)  # (B, len, 1)

        # 加 mask
        x_attn = x_attn.masked_fill(x_mask.unsqueeze(2) == 0, -1e9)

        # softmax，防 NaN
        x_attn = F.softmax(x_attn, dim=1)
        x_attn = torch.nan_to_num(x_attn, nan=0.0, posinf=1.0, neginf=0.0)

        out = (x * x_attn).sum(1)
        out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)

        return out

