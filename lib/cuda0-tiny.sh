# #!/bin/bash

betas=(0.5)
sample_fractions=(0.1)
algos=(fedavg)
max_jobs=1
lr=(0.1)
mu=(0.1)
# dir=(/mnt/data3/tiny-imagenet-200/)

current_jobs=0

for beta in "${betas[@]}"; do
    for sample_fraction in "${sample_fractions[@]}"; do
        for algo in "${algos[@]}"; do
            echo "algo=${algo}, sample_fractionrithm=${sample_fraction}, beta=${beta} 으로 실행 중"
            python3 main.py --device cuda:0 --datadir ${dir} --mu ${mu} --alg ${algo} --lr ${lr} --sample_fraction ${sample_fraction} --beta ${beta} --batch-size 50 --comm_round 100 --n_parties 100 --dataset cifar10 --model resnet18 --epochs 5 --partition noniid --reg 0.001 --save_model 1&
            current_jobs=$((current_jobs + 1))

            if [ "$current_jobs" -ge "$max_jobs" ]; then
                wait -n
                current_jobs=$((current_jobs - 1))
            fi
        done
    done
done

wait
echo "모든 작업이 완료되었습니다."