# MGTM

Official implementation of **MGTM: Adaptive Sparse Temporal Memory for Cross-Session Sticker Retrieval**.

MGTM is short for Multi-Granularity Temporal Memory Model. It uses ASTRA, the Adaptive Sparse Temporal Retrieval and Aggregation module, to read memory through Adaptive Multi-scale Temporal Decay (AMTD) and Dynamic Sparse Memory Aggregation (DSMA).

## Base weights

Download the Taiyi-CLIP text model and PVTv2-B2 weights from their original providers. Use the repository-relative layout below unless you plan to pass other locations to the training scripts.

```text
pretrained_weights/
|-- Taiyi-CLIP/
|   |-- config.json
|   |-- tokenizer files
|   `-- model weights
`-- PVT_v2_b2.pth
```

## Data layout

Obtain each benchmark under its own terms and keep the data outside Git tracking. All default paths are relative to the repository root.

### MultiChat

The public training interface accepts the `kuaile` and `yongyuan` modes.

```text
data/MultiChat/
|-- kuaile/intent.txt
|-- yongyuan/intent.txt
|-- kuaile_sample_intent_style_attribute.json
|-- yongyuan_sample_intent_style_attribute.json
|-- lmdb_kuaile_intent_style_attribute/
|   |-- train/pairs/
|   |-- val/pairs/
|   |-- test/pairs/
|   |-- train_imgs/
|   |-- val_imgs/
|   `-- test_imgs/
`-- lmdb_yongyuan_intent_style_attribute/
    `-- same split layout as above
```

Each `pairs/` or `*_imgs/` entry is an LMDB directory.

### DSTC10-MOD

```text
data/DSTC10-MOD/
|-- dialogues.jsonl
|-- splits/
|   |-- train.jsonl
|   |-- val.jsonl
|   `-- test.jsonl
|-- images/
|   `-- <sticker_id>.png
`-- labels/
    `-- sticker_emotion_top2_train.json
```

## Training

### MultiChat

The following command trains MGTM on `kuaile`.

```bash
python scripts/train_multichat.py --data-mode kuaile --data-root data/MultiChat --text-model pretrained_weights/Taiyi-CLIP --pvt-weights pretrained_weights/PVT_v2_b2.pth --output-root outputs --batch-size 16 --lr 1e-4
```

Use the same command with `--data-mode yongyuan` for the other MultiChat mode.

### DSTC10-MOD

```bash
python scripts/train_dstc10_mod.py --data-root data/DSTC10-MOD --taiyi-text-model pretrained_weights/Taiyi-CLIP --pvt-weights pretrained_weights/PVT_v2_b2.pth --output-root outputs --batch-size 32 --lr 1e-4
```

## License

The source code is released under the MIT License. MultiChat, DSTC10-MOD, Taiyi-CLIP, and PVTv2 weights remain subject to their respective licenses.

Publication metadata will be added when it is available. Until then, cite the paper by its title and include a link to this repository.
