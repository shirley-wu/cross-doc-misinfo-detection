import random

import dgl
import numpy as np
import sklearn.metrics
import torch

from models import Heterogeneous_GNN, Fuse

MODEL_CLS_DICT = {
    'hetero': Heterogeneous_GNN,
    'fuse': Fuse,
}


def add_shared_args(parser):
    # model
    parser.add_argument('--model', default='hetero', choices=MODEL_CLS_DICT.keys())
    parser.add_argument('--n-heads', default=8, type=int)
    parser.add_argument('--n-layers', default=4, type=int)
    # fuse model
    parser.add_argument('--feature-ckpt')
    parser.add_argument('--random-features', action="store_true")

    # data
    parser.add_argument('--ablation', default=None, choices=['event-coref', ])
    parser.add_argument('--event-labels-version', default=None)

    parser.add_argument('--seed', default=42, type=int)

    return parser


def get_best_th(labels, probs, step=0.005):
    labels = np.array(labels)

    thresholds = np.arange(0, 1 + step, step)
    preds = (probs[:, None] >= thresholds[None, :])
    tp = (labels[:, None] * preds).sum(axis=0)
    recall = tp / labels.sum()
    prec = tp / preds.sum(axis=0)
    f1 = 2 * prec * recall / (prec + recall)
    f1[np.isnan(f1)] = -1

    return thresholds[np.argmax(f1)]


def calc_metrics(labels_all, logits_all, threshold=None, return_best_th=False):
    # probs
    probs_all = np.exp(logits_all)
    probs_all /= probs_all.sum(axis=1).reshape(-1, 1)

    # acc, f1, precision, recall (based on pred v.s. label)
    if threshold is None:
        threshold = 0.5
    preds_all = probs_all[:, 1] > threshold
    acc = sklearn.metrics.accuracy_score(labels_all, preds_all)
    f1 = sklearn.metrics.f1_score(labels_all, preds_all)
    precision = sklearn.metrics.precision_score(labels_all, preds_all)
    recall = sklearn.metrics.recall_score(labels_all, preds_all)

    # auc
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(labels_all, probs_all[:, 1])
    auc = sklearn.metrics.auc(fpr, tpr)

    # best f1 threshold
    best_th = get_best_th(labels_all, probs_all[:, 1]) if return_best_th else None

    return dict(acc=acc * 100, precision=precision * 100, recall=recall * 100, f1=f1 * 100, best_th=best_th,
                n_pred=sum(preds_all), n_gt=sum(labels_all), auc=auc * 100, logits=logits_all, )


def eval_two_task(dataloader_val, model, loss_d_fn, loss_e_fn, threshold=None, **kwargs):
    if threshold is None:
        th_d = th_e = None
    else:
        th_d, th_e = threshold

    with torch.no_grad():
        model.eval()

        logits_d_all = []
        labels_d_all = []
        losses_d = []
        logits_e_all = []
        labels_e_all = []
        losses_e = []

        for idx, inputs in enumerate(dataloader_val):
            preds_d, labels_d, preds_e, labels_e = model(inputs)

            if loss_d_fn is not None:
                loss_d = loss_d_fn(preds_d, labels_d)
                losses_d.append(float(loss_d))
            logits_d_all.append(preds_d.cpu().numpy())
            labels_d_all += labels_d.tolist()

            if loss_e_fn is not None:
                loss_e = loss_e_fn(preds_e, labels_e)
                losses_e.append(float(loss_e))
            logits_e_all.append(preds_e.cpu().numpy())
            labels_e_all += labels_e.tolist()

        logits_d_all = np.concatenate(logits_d_all, axis=0)
        logits_e_all = np.concatenate(logits_e_all, axis=0)
        metrics_d = calc_metrics(labels_d_all, logits_d_all, threshold=th_d, **kwargs)
        metrics_e = calc_metrics(labels_e_all, logits_e_all, threshold=th_e, **kwargs)

        loss_d = np.mean(losses_d) if loss_d_fn is not None else None
        loss_e = np.mean(losses_e) if loss_e_fn is not None else None
        return metrics_d, loss_d, metrics_e, loss_e


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    dgl.seed(seed)


def count_event_fake_ratio(graphs):
    labels = []
    for g in graphs:
        labels += g.nodes['event'].data['labels'].tolist()
    return sum(labels) / len(labels)
