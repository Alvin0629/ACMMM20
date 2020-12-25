# ACMMM20




First train the view generation network, we incoporate 2D-3D-2D concept to improve gan-based pixel-level synthesis:

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
python train.py 
    --net marrnet2 \
    --dataset shapenet \
    --classes "$class" \
    --canon_sup \
    --batch_size 8 \
    --epoch_batches 2500 \
    --eval_batches 5 \
    --optim adam \
    --lr 1e-3 \
    --epoch 200 \
    --vis_batches_vali 50 \
    --gpu "$gpu" \
    --save_net 10 \
    --workers 4 \
    --logdir "$outdir" \
    --suffix '{classes}_canon-{canon_sup}' \
    --tensorboard \
```

