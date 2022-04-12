import copy
import gc
import json
import os
from collections import defaultdict

import dgl
import lxml.etree as ET
import torch
import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer

NTYPES = ['doc', 'ent', 'event', 'center', ]
ETYPES = [
    ('ent', 'rel', 'ent'), ('ent', 'rel-inv', 'ent'),
    ('ent', 'ent2doc', 'doc'), ('doc', 'doc2ent', 'ent'),
    ('event', 'arg', 'ent'), ('ent', 'arg-inv', 'event'),
    ('event', 'event2doc', 'doc'), ('doc', 'doc2event', 'event'),
    ('event', 'event2center', 'center'), ('center', 'center2event', 'event'),
]


def parse_offset_str(offset_str):
    doc_id = offset_str[:offset_str.rfind(':')]
    start = int(offset_str[offset_str.rfind(':') + 1:offset_str.rfind('-')])
    end = int(offset_str[offset_str.rfind('-') + 1:])
    if '-0__' in doc_id:
        print("Fix:", doc_id)
        doc_id = doc_id.replace('-0__', '-01__')
    return doc_id, start, end


def parse_entities(entity_cs):
    entities = dict()

    with open(entity_cs) as f:
        entity_lines = [x for x in f.read().strip().splitlines() if x.startswith(":Entity")]
    for line in entity_lines:
        line = line.strip().split('\t')
        if line[1].endswith("mention") and line[1] != "canonical_mention":
            doc_id, _, _ = parse_offset_str(line[3])
            entity_uniq_id = '{}__{}'.format(doc_id, line[0])
            if entity_uniq_id not in entities:
                entities[entity_uniq_id] = dict(doc_id=doc_id, entity_id=line[0], mention=set())
            entities[entity_uniq_id]['mention'].add(line[2][1:-1])

    for entity_uniq_id in entities:
        mentions = sorted(entities[entity_uniq_id]['mention'], key=lambda x: len(x), reverse=True)
        entities[entity_uniq_id]['mention'] = mentions[0]

    return entities


def parse_relations(rel_cs):
    relations = defaultdict(set)

    with open(rel_cs) as f:
        lines = f.read().strip().splitlines()
    for i, line in enumerate(lines):
        label_from, type_rel, label_to, offset, _ = line.strip().split("\t")
        type_rel = type_rel.strip().split("#")[-1]
        doc_id, _, _ = parse_offset_str(offset)
        label_from = '{}__{}'.format(doc_id, label_from)
        label_to = '{}__{}'.format(doc_id, label_to)
        relations[label_from].add((type_rel, label_to))

    return relations


def parse_events(event_cs):
    with open(event_cs) as f:
        lines = [line for line in f if line.startswith("::Event")]

    events_lines = []
    for line in lines:
        if line.rstrip('\n').split('\t')[1] == 'type':
            events_lines.append([])
        events_lines[-1].append(line)

    events = dict()
    events_args = defaultdict(set)

    for elines in events_lines:
        tabs = [line.rstrip('\n').split('\t') for line in elines]
        # event id & type
        assert tabs[0][1] == 'type'
        event_id = tabs[0][0]
        event_type = tabs[0][2].split("#")[-1]
        # mention
        mentions = [t[3] for t in tabs if 'mention' in t[1]]
        assert len(set(mentions)) == 1, "Multiple mentions for event %s (mentions = %r) in file %s" % (
            event_id, mentions, event_cs
        )
        event_mention = mentions[0]
        doc_id, _, _ = parse_offset_str(event_mention)
        # event uniq id
        event_uniq_id = '{}__{}'.format(event_id, event_mention)
        events[event_uniq_id] = dict(doc_id=doc_id, event_type=event_type, event_id=event_id, mention=event_mention)
        # args
        for t in tabs:
            if 'mention' not in t[1] and '_' in t[1]:
                assert doc_id == parse_offset_str(t[3])[0]
                typestr = t[1]
                arg_role = typestr[typestr.rfind('_') + 1:].split('.')[0]
                arg_id = '{}__{}'.format(doc_id, t[2])
                events_args[event_uniq_id].add((arg_role, arg_id))

    return events, events_args


def filter_entities(events_args, entities):
    entities_appear = set()
    for v in events_args.values():
        for _, e in v:
            entities_appear.add(e)
    ret = {k: v for k, v in entities.items() if k in entities_appear}
    return ret


def filter_relations(entities, relations):
    ret = {}
    for k, v in relations.items():
        if k in entities:
            v_ = [(predicate, obj) for predicate, obj in v if obj in entities]
            if len(v_) > 0:
                ret[k] = v_
    return ret


def parse_cs(entity_cs, rel_cs, event_cs):
    entities = parse_entities(entity_cs)
    relations = parse_relations(rel_cs)
    events, events_args = parse_events(event_cs)
    entities = filter_entities(events_args, entities)
    relations = filter_relations(entities, relations)
    return entities, relations, events, events_args


def read_doc_text(tree):
    doc_tokens = []
    for seg in tree.find('DOC').find('TEXT').findall('SEG'):
        doc_tokens.append(seg.find('ORIGINAL_TEXT').text)
    return ' '.join(doc_tokens)


def read_event_text(tree, start, end):
    for seg in tree.find('DOC').find('TEXT').findall('SEG'):
        seg_start = int(seg.get('start_char'))
        seg_end = int(seg.get('end_char'))
        if seg_start <= start and end <= seg_end:
            return seg.find('ORIGINAL_TEXT').text
    raise ValueError("Invalid start %d end %d" % (start, end))


class ClusterDataset(Dataset):
    def __init__(self, root, split_file, pretrained, ablation=None):
        self.root = root
        self.split_file = split_file
        self.data = []
        self.ablation = ablation

        self.pretrained = pretrained
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained)

        with open(split_file) as f:
            for line in f:
                line = json.loads(line)
                cs_dir = os.path.join(root, 'merged', 'cluster-{:d}'.format(line["cluster_id"]))
                kg = parse_cs(
                    os.path.join(cs_dir, 'entity.cs'),
                    os.path.join(cs_dir, 'relation.cs'),
                    os.path.join(cs_dir, 'event.cs'),
                )
                self.data.append((line['documents'], kg))

        with open(os.path.join(self.root, 'event_labels.json')) as f:
            self.event_labels = json.load(f)

        self.graphs = None
        if len(self.data) <= 500:
            precache_bert_files = [(0, len(self.data), '{:s}.precache_bert.pkl'.format(split_file)), ]
        else:
            precache_bert_files = []
            for s in range(0, len(self.data), 500):
                e = min(s + 500, len(self.data))
                precache_bert_files.append((s, e, '{:s}.precache_bert_{:d}-{:d}.pkl'.format(split_file, s, e)))

        data = []
        graphs = []
        for s, e, fname in precache_bert_files:
            if os.path.exists(fname):
                data_, graphs_ = torch.load(fname)
            else:
                data_ = self.data[s:e]
                graphs_ = self.get_bert_graphs(s, e)
                torch.save((data_, graphs_), fname)
            data += data_
            graphs += graphs_
        self.data = data
        self.graphs = graphs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        g = self.graphs[idx]
        if self.ablation == 'event-coref':
            for etype in [('event', 'event2center', 'center'), ('center', 'center2event', 'event'), ]:
                g.remove_edges(torch.arange(g.num_edges(etype)).to(g.device), etype=etype)
        return g

    def get_bert_graphs(self, start, end, bsz: int = 8192):
        graphs = []

        bert = AutoModel.from_pretrained(self.pretrained).cuda()

        with torch.no_grad():

            def bert_encode(data):
                with torch.no_grad():
                    n, l = data['input_ids'].shape
                    if n * l > bsz:
                        n_each = max(bsz // l, 1)
                    else:
                        n_each = max(1, l)
                    h = []
                    for i in range(0, n, n_each):
                        h_ = bert(
                            input_ids=data['input_ids'][i:i + n_each],
                            attention_mask=data['attention_mask'][i:i + n_each],
                        )["pooler_output"]
                        h.append(h_)
                    if len(h) == 0:
                        h = torch.zeros((0, 768)).cuda()
                    else:
                        h = torch.cat(h, dim=0)
                return h

            for gi in tqdm.trange(start, end, desc='cache ' + os.path.basename(self.split_file)):
                g = self.construct_dgl_graph(self.data[gi])
                g = g.to('cuda')

                for t in NTYPES:
                    if 'input_ids' in g.nodes[t].data:
                        g.nodes[t].data['x'] = bert_encode({
                            k: v for k, v in g.nodes[t].data.items() if k in ['input_ids', 'attention_mask', ]
                        })
                    else:
                        assert t == "center"
                        g.nodes[t].data['x'] = torch.zeros((g.num_nodes(t), 768), dtype=torch.float, device='cuda')

                for t in ETYPES:
                    if 'input_ids' in g.edges[t].data:
                        g.edges[t].data['x'] = bert_encode({
                            k: v for k, v in g.edges[t].data.items() if k in ['input_ids', 'attention_mask', ]
                        })
                    else:
                        g.edges[t].data['x'] = torch.zeros((g.num_edges(t), 768), dtype=torch.float, device='cuda')

                graphs.append(copy.deepcopy(g.to('cpu')))
                del g
                gc.collect()
                torch.cuda.empty_cache()

        return graphs

    def construct_dgl_graph(self, data):
        g = {}
        doc_ids, doc_labels = zip(*data[0])
        entities, relations, events, events_args = data[1]

        trees = [ET.parse(os.path.join(self.root, 'ltf', '{}.ltf.xml'.format(doc_id))) for doc_id in doc_ids]

        doc_features = self.construct_doc_nodes(trees, doc_labels)
        # g[('doc', 'doc-selfloop', 'doc')] = (torch.arange(len(doc_ids)), torch.arange(len(doc_ids)))

        ent_int2id, ent_id2int, ent_features = self.construct_ent_nodes(entities)
        # g[('ent', 'ent-selfloop', 'ent')] = (torch.arange(len(entities)), torch.arange(len(entities)))
        rel_features, rel_edges = self.construct_rel_edges(entities, relations, ent_id2int)
        g[('ent', 'rel', 'ent')] = (rel_edges[:, 0], rel_edges[:, 1])
        g[('ent', 'rel-inv', 'ent')] = (rel_edges[:, 1], rel_edges[:, 0])
        ent2doc_edges = self.construct_supernode_edges(
            ent_int2id, lambda x: '__'.join(x.split("__")[:-1]).replace('-0__', '-01__'), doc_ids
        )
        g[('ent', 'ent2doc', 'doc')] = (ent2doc_edges[:, 0], ent2doc_edges[:, 1])
        g[('doc', 'doc2ent', 'ent')] = (ent2doc_edges[:, 1], ent2doc_edges[:, 0])

        event_int2id, event_id2int, event_features = self.construct_event_nodes(events, doc_ids, trees)
        # g[('event', 'event-selfloop', 'event')] = (torch.arange(len(events)), torch.arange(len(events)))
        arg_features, arg_edges = self.construct_arg_edges(events, events_args, event_id2int, entities, ent_id2int)
        g[('event', 'arg', 'ent')] = (arg_edges[:, 0], arg_edges[:, 1])
        g[('ent', 'arg-inv', 'event')] = (arg_edges[:, 1], arg_edges[:, 0])
        event2doc_edges = self.construct_supernode_edges(
            event_int2id, lambda x: '__'.join(x.split("__")[1:]).split(":")[0].replace('-0__', '-01__'), doc_ids
        )
        g[('event', 'event2doc', 'doc')] = (event2doc_edges[:, 0], event2doc_edges[:, 1])
        g[('doc', 'doc2event', 'event')] = (event2doc_edges[:, 1], event2doc_edges[:, 0])

        center_id2int = self.construct_event_center_nodes(event_int2id)
        # g[('center', 'center-selfloop', 'center')] = (
        #     torch.arange(len(center_id2int)), torch.arange(len(center_id2int)))
        event_coref_edges = self.construct_event_center_coref_edges(event_int2id, center_id2int)
        g[('event', 'event2center', 'center')] = (event_coref_edges[:, 0], event_coref_edges[:, 1])
        g[('center', 'center2event', 'event')] = (event_coref_edges[:, 1], event_coref_edges[:, 0])

        g = dgl.heterograph(g, num_nodes_dict={'doc': len(doc_ids), 'ent': len(entities), 'event': len(events),
                                               'center': len(center_id2int), })

        g.nodes['doc'].data.update(doc_features)
        g.nodes['ent'].data.update(ent_features)
        g.nodes['event'].data.update(event_features)

        g.edges['rel'].data.update(rel_features)
        g.edges['rel-inv'].data.update(rel_features)
        g.edges['arg'].data.update(arg_features)
        g.edges['arg-inv'].data.update(arg_features)

        return g

    def construct_doc_nodes(self, trees, doc_labels):
        # Construct doc super nodes
        doc_texts = [read_doc_text(tree) for tree in trees]
        doc_features = self.encode_strings_to_features(doc_texts, max_length=512)
        doc_features['labels'] = torch.LongTensor(doc_labels)
        return doc_features

    def construct_ent_nodes(self, entities):
        ent_int2id = sorted(entities.keys())
        ent_id2int = {x: i for i, x in enumerate(ent_int2id)}
        ent_texts = [entities[n]['mention'] for n in ent_int2id]
        ent_features = self.encode_strings_to_features(ent_texts, max_length=30)
        return ent_int2id, ent_id2int, ent_features

    def construct_rel_edges(self, entities, relations, ent_id2int):
        rels = []
        rel_texts = []
        for subj in relations:
            for predicate, obj in relations[subj]:
                rels.append((ent_id2int[subj], ent_id2int[obj]))
                rel_texts.append("{}, {}, {}".format(
                    entities[subj]['mention'], predicate.replace('.', ' ').strip(), entities[obj]['mention'],
                ))
        rel_features = self.encode_strings_to_features(rel_texts, max_length=100)
        rels = torch.LongTensor(rels).reshape(-1, 2)
        return rel_features, rels

    def construct_event_nodes(self, events, doc_ids, trees):
        # Construct event nodes
        event_int2id = sorted(events.keys())
        event_id2int = {x: i for i, x in enumerate(event_int2id)}
        event_texts = []
        for event_id in event_int2id:
            doc_id, start, end = parse_offset_str(events[event_id]['mention'])
            event_texts.append(read_event_text(trees[doc_ids.index(doc_id)], start, end))
        event_features = self.encode_strings_to_features(event_texts, max_length=256)
        event_labels = [self.event_labels[event_id] for event_id in event_int2id]
        event_features['labels'] = torch.LongTensor(event_labels)
        return event_int2id, event_id2int, event_features

    def construct_arg_edges(self, events, events_args, event_id2int, entities, ent_id2int):
        args = []
        arg_texts = []
        for event in events_args:
            for role, ent in events_args[event]:
                args.append((event_id2int[event], ent_id2int[ent]))
                arg_texts.append("{}, {}, {}".format(
                    events[event]['event_type'].replace('.', ' ').strip(), role, entities[ent]['mention'],
                ))
        arg_features = self.encode_strings_to_features(arg_texts, max_length=100)
        args = torch.LongTensor(args).reshape(-1, 2)
        return arg_features, args

    def construct_event_coref_edges(self, event_int2id):
        event_clusters = defaultdict(set)
        for i, event_id in enumerate(event_int2id):
            event_clusters[event_id.split("__")[0]].add(i)
        edges = []
        for values in event_clusters.values():
            for i in values:
                for j in values:
                    if i != j:
                        edges.append((i, j))
        return torch.LongTensor(edges).reshape(-1, 2)

    def construct_event_center_nodes(self, event_int2id):
        center_id2int = dict()
        for event_id in event_int2id:
            event_cluster_id = event_id.split("__")[0]
            if event_cluster_id not in center_id2int:
                center_id2int[event_cluster_id] = len(center_id2int)
        return center_id2int

    def construct_event_center_coref_edges(self, event_int2id, center_id2int):
        edges = []
        for i, event_id in enumerate(event_int2id):
            event_cluster_id = event_id.split("__")[0]
            edges.append((i, center_id2int[event_cluster_id]))
        return torch.LongTensor(edges).reshape(-1, 2)

    def construct_supernode_edges(self, node_int2id, get_doc_id_fn, doc_ids):
        node2doc = []
        for i, node_id in enumerate(node_int2id):
            doc_id = get_doc_id_fn(node_id)
            node2doc.append((i, doc_ids.index(doc_id)))
        return torch.LongTensor(node2doc).reshape(-1, 2)

    def encode_strings_to_features(self, strings, max_length=30):
        if len(strings) == 0:
            return dict(input_ids=torch.zeros((0, 0)).long(), attention_mask=torch.zeros((0, 0)).long())
        ret = self.tokenizer(strings, padding=True, truncation=True, max_length=max_length)
        return dict(
            input_ids=torch.LongTensor(ret['input_ids']),
            attention_mask=torch.LongTensor(ret['attention_mask']),
        )


def build_dataloader(args, split):
    dataset = ClusterDataset(
        args.root, os.path.join(args.root, "split.{}.json".format(split)), 'bert-base-uncased', ablation=args.ablation,
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=(split == 'train'), collate_fn=lambda x: x,
    )
    return dataloader
