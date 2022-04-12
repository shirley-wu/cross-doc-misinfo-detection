import argparse

import torch

from data import build_dataloader
from utils import add_shared_args, eval_two_task, set_seed, MODEL_CLS_DICT


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('root')
    parser.add_argument('ckpt')
    parser.add_argument('--set', default="test", choices=["valid", "test", ])

    parser = add_shared_args(parser)

    parser.add_argument('--metrics', default=['acc', 'precision', 'recall', 'f1', 'auc', 'n_pred', 'n_gt', ])
    parser.add_argument('--use-th', default=None)

    # data loading paras
    parser.add_argument('--batch-size', default=1, type=int)

    # dump parameters
    parser.add_argument('--no-write', action="store_true")
    parser.add_argument('--dump-best-th', action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(args)

    set_seed(args.seed)

    dataloader = build_dataloader(args, args.set)
    model = MODEL_CLS_DICT[args.model](args)
    model = model.cuda()
    print(model)

    state_dict = torch.load(args.ckpt)
    model.load_state_dict(state_dict)

    if args.use_th:
        assert args.set == "test"
        threshold = torch.load(args.use_th)
    else:
        threshold = None

    metrics_d, _, metrics_e, _ = eval_two_task(dataloader, model, None, None, threshold=threshold,
                                               return_best_th=args.dump_best_th)

    LOG_S = "Set {:s}:\nDoc:".format(args.set)
    for k in args.metrics:
        LOG_S += " {:s} = {:.2f},".format(k, metrics_d[k])
    LOG_S = LOG_S[:-1] + "\nEvent:"
    for k in args.metrics:
        LOG_S += " {:s} = {:.2f},".format(k, metrics_e[k])
    LOG_S = LOG_S[:-1]

    print(LOG_S)

    if not args.no_write:
        with open(f"{args.ckpt}.{args.set}.out", "w") as f:
            f.write(LOG_S + '\n')
        torch.save(metrics_d['logits'], f"{args.ckpt}.{args.set}.doc-logits.pkl")
        torch.save(metrics_e['logits'], f"{args.ckpt}.{args.set}.event-logits.pkl")

    if args.dump_best_th:
        assert args.set == "valid"
        print("Doc best th = %.2f, event best th = %.2f" % (metrics_d['best_th'], metrics_e['best_th']))
        torch.save((metrics_d['best_th'], metrics_e['best_th']), f"{args.ckpt}.{args.set}.best-th.pkl")
