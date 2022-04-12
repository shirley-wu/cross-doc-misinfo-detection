This repository is for our NAACL2022 paper: Cross-document Misinformation Detection based on Event Graph Reasoning

## Requirements

1. Install requirements using `pip install -r requirements.txt`
2. Install `dgl==0.6.1` following the [instructions](https://www.dgl.ai/pages/start.html)

## Dataset

Please download data from [Google Drive](https://drive.google.com/file/d/1YwgqNi8yQMDuXSEvlB8-TT8o92X1vGQk/view?usp=sharing).
The file contains 3 directories (IED, TL17, Crisis), each for one dataset.
Each directory is organized as follows:

    {IED,TL17,Crisis}
    ├── split.{train,valid,test}.json    # Split and labels for training / valid / test sets
    ├── event_labels.json                # Labels for event-level detection
    ├── ltf/                             # Tokenized documents in xml format
    └── merged/                          # IE outputs for clusters
        ├── cluster-0/                     # IE outputs for cluster 0 
        │   ├── edl.json                     # Entity linking results
        │   └── {entity,relation,event}.cs   # IE outputs of entity / relation / event extraction in cold-start format
        └── ...

## Training

Specify `$DATA $OUTPUT $EPOCH` before you continue.
`$EPOCH` for IED, TL17 and Crisis datasets are 120, 120 and 10 respectively.

* Train event-level detector:
```
python train.py $DATA $OUTPUT --accum-step 16 --lr 5e-05 --num-epochs $EPOCH --warmup 100 \
  --grad-clip 1.0 --scheduler linear-warmup --model hetero --event-detection-lambda 1.0
```
* Train document-level detector:
```
python train.py $DATA $OUTPUT --accum-step 16 --lr 5e-05 --num-epochs $EPOCH --warmup 100 \
  --grad-clip 1.0 --scheduler linear-warmup --model hetero --event-detection-lambda 0.0
```
* Train document-level detector using event-level features: (`$EVENT_CKPT` is checkpoint for the event-level detector)
```
python train.py $DATA $OUTPUT --accum-step 16 --lr 5e-05 --num-epochs $EPOCH --warmup 100 \
  --grad-clip 1.0 --scheduler linear-warmup --model fuse --event-detection-lambda 0.0 \
  --feature-ckpt $EVENT_CKPT
```

## Evaluation

Specify `$DATA $OUTPUT` before you continue.

* Evaluate document-level detector:
```
python eval.py $DATA $OUTPUT/checkpoint-best.doc.pt --model hetero --set test --event-detection
```
* Evaluate document-level detector using event-level features: (`$EVENT_CKPT` is checkpoint for the event-level detector)
```
python eval.py $DATA $OUTPUT/checkpoint-best.doc.pt --model fuse --feature-ckpt $EVENT_CKPT --set test --event-detection
```
* Evaluate event-level detector. Since event-level detection data is too imbalanced, we first select threshold based on the valid set, and then evalute on the test set:
```
python eval.py $DATA $OUTPUT/checkpoint-best.event.pt --model hetero --set valid --event-detection --dump-best-th
python eval.py $DATA $OUTPUT/checkpoint-best.event.pt --model hetero --set test --event-detection \
  --use-th $OUTPUT/checkpoint-best.event.pt.valid.best-th.pkl
```