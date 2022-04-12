import dgl
import numpy as np
import torch
from dgl import function as fn
from torch import nn


class EdgeGATConv(nn.Module):
    def __init__(self, in_feats, out_feats, num_heads, feat_drop=0., attn_drop=0., negative_slope=0.2,
                 allow_zero_in_degree=False):
        super(EdgeGATConv, self).__init__()
        self._allow_zero_in_degree = allow_zero_in_degree

        self._num_heads = num_heads
        self._in_src_feats = self._in_dst_feats = in_feats
        self._out_feats = out_feats
        self.scaler = 1 / np.sqrt(self._out_feats)

        self.fc_q = nn.Linear(self._in_src_feats, out_feats * num_heads)
        self.fc_r = nn.Linear(self._in_src_feats * 3, out_feats * num_heads)
        self.fc_k = nn.Linear(out_feats * num_heads, out_feats * num_heads)
        self.fc_v = nn.Linear(out_feats * num_heads, out_feats * num_heads)
        self.fc_out = nn.Linear(out_feats * num_heads, out_feats * num_heads)

        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.postnorm_node = torch.nn.BatchNorm1d(in_feats)

        self.reset_parameters()

    def reset_parameters(self):
        def init_fc(fc):
            nn.init.xavier_normal_(fc.weight, gain=1.0)
            nn.init.constant_(fc.bias, 0)

        init_fc(self.fc_q)
        init_fc(self.fc_r)
        init_fc(self.fc_k)
        init_fc(self.fc_v)
        init_fc(self.fc_out)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, node_feat, edge_feat):  # , get_attention=False):
        def calc_relation_emb(edges):
            x = torch.cat([edges.src['x'], edges.dst['x'], edges.data['x'], ], dim=-1)
            return {'r': self.fc_r(x), }

        with graph.local_scope():
            if not self._allow_zero_in_degree:
                assert not (graph.in_degrees() == 0).any()

            if isinstance(node_feat, tuple):
                src_feat, dst_feat = node_feat
            else:
                src_feat = dst_feat = node_feat
            residual = dst_feat

            src_feat = self.feat_drop(src_feat)
            dst_feat = self.feat_drop(dst_feat)
            edge_feat = self.feat_drop(edge_feat)

            graph.srcdata.update({'x': src_feat, })
            graph.dstdata.update({'x': dst_feat, })
            graph.edata.update({'x': edge_feat, })
            # calc r
            graph.apply_edges(calc_relation_emb)
            r = graph.edata['r']
            # calc k, q, v
            q = self.fc_q(dst_feat)
            k = self.fc_k(r)
            v = self.fc_v(r)
            # split head
            q = q.view(*q.shape[:-1], self._num_heads, self._out_feats)
            k = k.view(*k.shape[:-1], self._num_heads, self._out_feats)
            v = v.view(*v.shape[:-1], self._num_heads, self._out_feats)
            # compute edge attention
            graph.dstdata.update({'q': q, })
            graph.edata.update({'k': k, })
            graph.apply_edges(fn.v_mul_e('q', 'k', 'a'))
            a = graph.edata['a'].sum(dim=-1) * self.scaler
            # compute softmax
            a = self.attn_drop(dgl.nn.softmax.edge_softmax(graph, a))
            # message passing
            m = v * a.unsqueeze(-1)
            graph.edata.update({'m': m, })
            graph.update_all(fn.copy_edge('m', 'm'), fn.sum('m', 'out'))
            out = graph.dstdata['out']
            # out projection
            out = self.fc_out(out.flatten(1))
            # residual
            out = residual + out
            out = self.postnorm_node(out)

            return out
