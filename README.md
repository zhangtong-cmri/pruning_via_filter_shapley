# pruning_via_filter_shapley
**Unofficial PyTorch implementation of pruning VGG and resnet on CIFAR-10 Data set**
# Example Scripts
**Train Resnet20 on CIFAR-10 Data set**
```
python main.py --is_train --origin --epoch 200
```
**Prune Resnet20 via filter shapley and Retrain**
```
python main.py --is_train --pruned_model_dir ./results/. --epoch 40
```
