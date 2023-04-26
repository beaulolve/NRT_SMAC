
## Run an experiment 

```shell
CUDA_VISIBLE_DEVICES=0 python3 main.py --config=NRT_QMIX --env-config=sc2_3m with state_vae_train_buffer=10240 state_vae_train_batch=512 seed=0
CUDA_VISIBLE_DEVICES=0 python3 main.py --config=NRT_QMIX --env-config=sc2_2m_vs_1z with seed=0
CUDA_VISIBLE_DEVICES=0 python3 main.py --config=NRT_QMIX --env-config=sc2_2s_vs_1sc with seed=0
```

