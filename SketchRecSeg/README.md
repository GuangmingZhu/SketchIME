# SketchRecSeg

## Requirments

- Pytorch>=1.6.0
- pytorch_geometric>=1.6.1
- tensorboardX>=1.9

## Get the 56k data

click [download](https://drive.google.com/drive/folders/15nCiPF1NEGJ5j7v_MsnKp-eydtvjm8oj?usp=sharing "data") to get the 56k data used in ACMMM2023


move the data to <strong>./data</strong> dir

## How to use
**Test**

```bash
python evaluate.py  --class-name sketchrecseg --out-segment 140 --timestamp BEST --which-epoch bestloss
```

**Train**

````bash
python train.py  --class-name sketchrecseg --out-segment 140 --shuffle --stochastic
````

## Get all 209k data

click [download](https://drive.google.com/drive/folders/1kcRXKWV4qMkT_B856lk53DkkRwUXeeJG?usp=sharing) to get all the 209k data that we constructed.
