# HyperKDMA: Distilling Recommender Systems via Hypernetwork-based Teacher Assistants

## Overview
This is the implementation for the paper [HyperKDMA: Distilling Recommender Systems via Hypernetwork-based Teacher Assistants](https://google.com). In this work, we propose HyperKDMA, a distillation scheme using multiple hypernetwork-based teacher assistants to bridge the teacher-student gap in knowledge distillation for top-K recommendation. We verify the effectiveness of our method through experiments using three base models: [BPR](https://https://arxiv.org/abs/1205.2618), [NeuMF](https://https://arxiv.org/abs/1708.05031) and [LightGCN](https://https://arxiv.org/abs/2002.02126); and two public data sets: [CiteULike](https://https://github.com/js05212/citeulike-t) and [Foursquare](https://https://sites.google.com/site/yangdingqi/home/foursquare-dataset).

## Examples
### Training teacher model
```
python3 main_no_KD --model BPR --dim 200 --dataset CiteULike 
```

### Training student model without KD
```
python3 main_no_KD --model BPR --dim 20 --dataset CiteULike 
```

### Training student model with KD using DE
```
python3 main_DE --model BPR --teacher_dim 200 --student_dim 20 --dataset CiteULike
```

### Training student model with KD using HyperKDME-DE 
```
python3 main_DETA --model BPR --teacher_dim 200 --student_dim 20 --num_TAs 8 --dataset CiteULike
```
