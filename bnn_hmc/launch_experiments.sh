#!/bin/sh

for batch_size in 1 5 10 20 50 100 200 250 500 1000 
do 
python3 run_vi.py --seed=10 --weight_decay=5. --dir=runs/vi/mnist/ \
  --dataset_name=mnist --model_name=mlp_classification \
  --init_step_size=1e-4 --num_epochs=50  --eval_freq=1 --batch_size=$batch_size \
  --save_freq=10 --optimizer=Adam \
  --vi_sigma_init=0.01 --temperature=1. --vi_ensemble_size=20 
done  
  

for batch_size in 1 5 10 20 50 100 200 250 500 1000 
do 
python3 run_sgmcmc.py --seed=15 --weight_decay=5 --dir=runs/sgmcmc/mnist/ \
  --dataset_name=mnist --model_name=mlp_classification --init_step_size=3e-9 \
  --num_epochs=100 --num_burnin_epochs=100 \
  --eval_freq=1 --batch_size=$batch_size --save_freq=100 \
  --momentum=0.9
done


for batch_size in 1 5 10 20 50 100 200 250 500 1000 
do 
python3 run_sgmcmc.py --seed=14 --weight_decay=5 --dir=runs/sgmcmc/mnist/ \
  --dataset_name=mnist --model_name=mlp_classification --init_step_size=3e-8 \
  --num_epochs=100 --num_burnin_epochs=100 \
  --eval_freq=1 --batch_size=$batch_size --save_freq=100 \
  --momentum=0.0
done
