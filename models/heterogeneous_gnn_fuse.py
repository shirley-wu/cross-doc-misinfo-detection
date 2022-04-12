"""
GAT adopted form https://github.com/dmlc/dgl/blob/0.7.x/python/dgl/nn/pytorch/conv/gatconv.py
Model borrowing QA-GNN: Reasoning with Language Models and Knowledge Graphs for Question Answering
"""
import torch
from dgl import function as fn
from torch import nn

from data import ETYPES
from models.heterogeneous_gnn import Heterogeneous_GNN

NTYPE_TO_FUSE = ["ent", "event", "center", ]


class Heterogeneous_GNN_Fuse(Heterogeneous_GNN):
    def __init__(self, args):
        super(Heterogeneous_GNN_Fuse, self).__init__(args)
        self.fuse_modules = nn.ModuleList([
            nn.ModuleDict({
                k: nn.Linear(768 * 2, 768) for k in NTYPE_TO_FUSE
            }) for _ in range(args.n_layers)
        ])

    def extract_features(self, g, exist_center_feature=False, features_to_fuse=None):
        if isinstance(g, list):
            assert len(g) == 1
            g = g[0].to('cuda')

        assert exist_center_feature
        assert features_to_fuse is not None

        node_feats = {t: g.nodes[t].data['x'] for t in ['doc', 'ent', 'event', 'center', ]}

        edge_feats = {t[1]: (g.edges[t].data['x'],) for t in ETYPES if 'input_ids' in g.edges[t].data}

        # representation
        h = node_feats
        # GNN forward
        for i, layer in enumerate(self.gnn):
            for ntype, fuse in self.fuse_modules[i].items():
                if ntype in features_to_fuse and ntype in h:
                    x = torch.cat([features_to_fuse[ntype], h[ntype], ], dim=-1)
                    h[ntype] = fuse(x)
            h = layer(g, h, edge_feats)

        return h


class Fuse(nn.Module):
    def __init__(self, args):
        super(Fuse, self).__init__()
        self.args = args

        if not args.random_features:
            self.feature = Heterogeneous_GNN(args)
            self.feature.load_state_dict(torch.load(args.feature_ckpt))
            for p in self.feature.parameters():
                p.requires_grad = False

        self.main = Heterogeneous_GNN_Fuse(args)

    def forward(self, graphs):
        assert len(graphs) == 1
        g = graphs[0].to('cuda')
        with torch.no_grad():
            g.update_all(fn.copy_u('x', 'm'), fn.mean('m', 'x'), etype='event2center')
            if self.args.random_features:
                h = {ntype: torch.randn_like(g.nodes[ntype].data['x']) for ntype in NTYPE_TO_FUSE}
            else:
                h = self.feature.extract_features(g, exist_center_feature=True)

        return self.main(g, exist_center_feature=True, features_to_fuse=h)
