
## Single Image Shape-from-Silhouettes 
### 28th ACM International Conference on Multimedia (ACM MM 20)

First train the view generation network, I incoporate 2D-3D-2D concept (borrowed from equivariant_neural_render) to improve gan-based pixel-level synthesis performance in original paper:

To train the network, use:
```
python3 experiments.py
```

To infer the output, use:
```
python3 infer.py
```

Then, the 3D generation network is able to be trained by:
```
python 3d_rec/train.py --net generator --dataset shapenet --classes "car" --batch_size 16 --epoch_batches 2500 --eval_batches 5  --optim adam --lr 1e-3 --epoch 200 --vis_batches_valid 50 --gpu 0 --save_net 10  --workers 4  --logdir "output" --suffix "{classes}_lr{lr}"
```

To test the reconstruction, use:

```
python 3d_rec/test.py --net generator --net_file "best.pt" --dataset shapenet --classes "car" --batch_size 1 --input_rgb "example.jpg" --input_mask "mask.jpg" --output_dir "result" --suffix "{net}"  --overwrite --workers 1 --gpu 0
```

The code is being updated and refined. If you summarize relevant works or refer to the code, please cite
```
@inproceedings{lu2020single,
  title={Single Image Shape-from-Silhouettes},
  author={Lu, Yawen and Wang, Yuxing and Lu, Guoyu},
  booktitle={Proceedings of the 28th ACM International Conference on Multimedia},
  pages={3604--3613},
  year={2020}
}
```

