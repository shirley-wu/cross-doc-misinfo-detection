"""
GAT adopted form https://github.com/dmlc/dgl/blob/0.7.x/python/dgl/nn/pytorch/conv/gatconv.py
Model borrowing QA-GNN: Reasoning with Language Models and Knowledge Graphs for Question Answering
"""
import dgl
import dgl.nn
import torch
import torch.nn.functional as F
from dgl import function as fn
from torch import nn

from models.edge_gatconv import EdgeGATConv


class GATConvWrap(nn.Module):
    def __init__(self, in_feats, out_feats, num_heads, **kwargs):
        super(GATConvWrap, self).__init__()
        self.gat = dgl.nn.GATConv(in_feats, out_feats, num_heads, **kwargs)
        D = out_feats * num_heads
        self.fc_out = nn.Linear(D, D)
        self.postnorm_node = nn.BatchNorm1d(D)

    def forward(self, *args, **kwargs):
        ret = self.gat(*args, **kwargs)
        assert isinstance(ret, torch.Tensor)
        ret = ret.reshape(ret.shape[0], -1)
        ret = self.fc_out(ret)
        ret = self.postnorm_node(ret)
        return ret


class Heterogeneous_GNN(nn.Module):
    def __init__(self, args):
        super(Heterogeneous_GNN, self).__init__()

        self.gnn = nn.ModuleList([])
        for _ in range(args.n_layers):
            mods = dict()
            for t in [('ent', 'rel', 'ent'), ('ent', 'rel-inv', 'ent'),
                      ('event', 'arg', 'ent'), ('ent', 'arg-inv', 'event'), ]:
                mods[t[1]] = EdgeGATConv(768, 768 // args.n_heads, args.n_heads, allow_zero_in_degree=True)
            for t in [('ent', 'ent2doc', 'doc'), ('doc', 'doc2ent', 'ent'),
                      ('event', 'event2doc', 'doc'), ('doc', 'doc2event', 'event'),
                      ('event', 'event2center', 'center'), ('center', 'center2event', 'event'), ]:
                mods[t[1]] = GATConvWrap(768, 768 // args.n_heads, args.n_heads,
                                         residual=True, allow_zero_in_degree=True)
            self.gnn.append(dgl.nn.HeteroGraphConv(mods))

        self.classify_g = nn.Sequential(
            nn.Linear(768, 512), nn.ReLU(), nn.Linear(512, 56), nn.ReLU(), nn.Linear(56, 2)
        )
        self.classify_e = nn.Sequential(
            nn.Linear(768, 512), nn.ReLU(), nn.Linear(512, 56), nn.ReLU(), nn.Linear(56, 2)
        )

        self.args = args

    def extract_features(self, g, exist_center_feature=False):
        if isinstance(g, list):
            assert len(g) == 1
            g = g[0].to('cuda')

        if not exist_center_feature:
            with torch.no_grad():
                g.update_all(fn.copy_u('x', 'm'), fn.mean('m', 'x'), etype='event2center')
        node_feats = {t: g.nodes[t].data['x'] for t in ['doc', 'ent', 'event', 'center', ]}

        edge_feats = {t: (g.edges[t].data['x'],) for t in ['rel', 'rel-inv', 'arg', 'arg-inv', ]}

        # representation
        h = node_feats
        # GNN forward
        for layer in self.gnn:
            h = layer(g, h, edge_feats)
        return h

    def forward(self, g, exist_center_feature=False, **kwargs):
        if isinstance(g, list):
            if len(g) > 1:
                assert not exist_center_feature
                for gg in g:
                    gg.ndata.pop('input_ids')
                    gg.ndata.pop('attention_mask')
                    gg.edata.pop('input_ids')
                    gg.edata.pop('attention_mask')
                g = dgl.batch(g).to('cuda')
            else:
                g = g[0].to('cuda')

        h = self.extract_features(g, exist_center_feature=exist_center_feature, **kwargs)

        doc_labels = g.ndata['labels']['doc']
        if 'doc' not in h:
            print("!" * 20)
            print("WARNING! doc not in h")
            h_doc = torch.zeros((1, 768)).cuda()
        else:
            h_doc = h['doc']
        out = self.classify_g(F.relu(h_doc))

        event_labels = g.ndata['labels']['event']
        if 'event' not in h:
            print("!" * 20)
            print("WARNING! event not in h")
            h_e = torch.zeros((0, 768)).cuda()
        else:
            h_e = h['event']
        out_e = self.classify_e(F.relu(h_e))
        return out, doc_labels, out_e, event_labels
