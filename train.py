import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import tqdm
from transformers import get_linear_schedule_with_warmup

from data import build_dataloader
from utils import add_shared_args, eval_two_task, set_seed, count_event_fake_ratio, calc_metrics, MODEL_CLS_DICT


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('root')
    parser.add_argument('output')

    parser = add_shared_args(parser)

    parser.add_argument('--main-metric', default='auc')
    parser.add_argument('--print-metrics', default=['f1', 'acc', ], nargs="+")

    # two tasks
    parser.add_argument('--event-detection-lambda', default=0.0, type=float)  # 0.0 for doc-level, 1.0 for event-level

    # training paras
    parser.add_argument('--lr', default=1e-05, type=float)
    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument('--num-epochs', default=12, type=int)
    parser.add_argument('--scheduler', default='linear-warmup', choices=['linear-warmup', ])
    parser.add_argument('--warmup', default=100, type=int)
    parser.add_argument('--grad-clip', default=1.0, type=float)
    parser.add_argument('--accum-step', default=2, type=int)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    print(args)
    os.makedirs(args.output, exist_ok=True)

    set_seed(args.seed)

    dataloader_val = build_dataloader(args, "valid")
    dataloader_train = build_dataloader(args, "train")
    model = MODEL_CLS_DICT[args.model](args)
    model = model.cuda()
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    assert args.scheduler == 'linear-warmup'
    assert args.warmup > 0
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup,
                                                num_training_steps=args.num_epochs * len(dataloader_train) // args.accum-step)
    best_doc_metric = - np.inf
    best_event_metric = - np.inf
    loss_d_fn = nn.CrossEntropyLoss(
        reduction="mean", weight=torch.FloatTensor([0.5, 0.5, ]).cuda()
    )
    event_fake_ratio = count_event_fake_ratio(dataloader_train.dataset)
    print("Event fake ratio = %.4f" % event_fake_ratio)
    loss_e_fn = nn.CrossEntropyLoss(
        reduction="mean", weight=torch.FloatTensor([event_fake_ratio, 1 - event_fake_ratio, ]).cuda()
    )

    idx = 0
    idx_at_epoch_start = 0
    for epoch in range(args.num_epochs):
        idx_at_epoch_start = idx
        logits_d_all, logits_e_all = [], []
        labels_d_all, labels_e_all = [], []
        losses_d, losses_e, losses = [], [], []
        loss_d_accum, loss_e_accum, loss_accum = 0.0, 0.0, 0.0

        pbar = tqdm.tqdm(dataloader_train, total=len(dataloader_train), desc="Epoch {:d}".format(epoch))
        for inputs in pbar:
            idx += 1

            logits_d, labels_d, logits_e, labels_e = model(inputs)
            loss_d = loss_d_fn(logits_d, labels_d)
            loss_e = loss_e_fn(logits_e, labels_e)
            loss = (1 - args.event_detection_lambda) * loss_d + args.event_detection_lambda * loss_e

            loss.backward()
            loss_d_accum += float(loss_d)
            loss_e_accum += float(loss_e)
            loss_accum += float(loss)

            logits_d_all += logits_d.tolist()
            labels_d_all += labels_d.tolist()
            logits_e_all += logits_e.tolist()
            labels_e_all += labels_e.tolist()

            if idx % args.accum_step == 0:
                if args.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                losses.append(float(loss_accum))
                loss_accum = 0
                losses_d.append(float(loss_d_accum))
                loss_d_accum = 0
                losses_e.append(float(loss_e_accum))
                loss_e_accum = 0

                pbar.set_postfix({
                    "loss": "{:.2f},{:.2f},{:.2f}".format(losses_d[-1], losses_e[-1], losses[-1]),
                    "doc_" + args.main_metric: "{:.2f}".format(
                        calc_metrics(labels_d_all, logits_d_all)[args.main_metric]),
                    "event_" + args.main_metric: "{:.2f}".format(
                        calc_metrics(labels_e_all, logits_e_all)[args.main_metric]),
                })

            if idx - idx_at_epoch_start == len(dataloader_train) - 1:
                LOG_S = "Epoch {:d} iter {:d}: train loss = {:.2f}".format(epoch, idx, np.mean(losses_d))
                train_d_metrics = calc_metrics(labels_d_all, logits_d_all)
                for k in [args.main_metric, ] + args.print_metrics:
                    LOG_S += ", {:s} = {:.2f}".format(k, train_d_metrics[k])
                LOG_S += ", event_loss = {:.2f}".format(np.mean(losses_e))
                train_e_metrics = calc_metrics(labels_e_all, logits_e_all)
                for k in [args.main_metric, ] + args.print_metrics:
                    LOG_S += ", event_{:s} = {:.2f}".format(k, train_e_metrics[k])

                doc_val_metrics, doc_loss, event_val_metrics, event_loss = eval_two_task(
                    dataloader_val, model, loss_d_fn, loss_e_fn
                )
                LOG_S += "; valid loss = {:.2f}".format(doc_loss * args.accum_step)
                for k in [args.main_metric, ] + args.print_metrics:
                    LOG_S += ", {:s} = {:.2f}".format(k, doc_val_metrics[k])
                LOG_S += ", event_loss = {:.2f}".format(event_loss * args.accum_step)
                for k in [args.main_metric, ] + args.print_metrics:
                    LOG_S += ", event_{:s} = {:.2f}".format(k, event_val_metrics[k])

                pbar.write(LOG_S)

                logits_d_all, logits_e_all = [], []
                labels_d_all, labels_e_all = [], []
                losses_d, losses_e, losses = [], [], []
                torch.save(model.state_dict(), os.path.join(args.output, 'checkpoint{:d}-{:d}.pt'.format(epoch, idx)))
                if doc_val_metrics[args.main_metric] > best_doc_metric:
                    pbar.write("=== Best doc %s: %.2f -> %.2f" % (
                        args.main_metric, best_doc_metric, doc_val_metrics[args.main_metric],
                    ))
                    best_doc_metric = doc_val_metrics[args.main_metric]
                    torch.save(model.state_dict(), os.path.join(args.output, 'checkpoint-best.doc.pt'))
                if event_val_metrics[args.main_metric] > best_event_metric:
                    pbar.write("=== Best event %s: %.2f -> %.2f" % (
                        args.main_metric, best_event_metric, event_val_metrics[args.main_metric],
                    ))
                    best_event_metric = event_val_metrics[args.main_metric]
                    torch.save(model.state_dict(), os.path.join(args.output, 'checkpoint-best.event.pt'))
