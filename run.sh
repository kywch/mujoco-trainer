for seed in $(seq 2 11)
do
    python cleanrl_ppo.py --train.seed $seed --track --wandb_group cleanrl
done