# #!/bin/bash

mus=(0.1 1 5 10)
betas=(0.5)
sample_fractions=(0.05)
algos=(feddec)
max_jobs=1
lrs=(0.1 0.01 0.001)
alpha_reverses=(0)
data=(tinyimagenet)
dir=(/mnt/data3/tiny-imagenet-200/)

current_jobs=0
for lr in "${lrs[@]}"; do
    for alpha in "${alpha_reverses[@]}"; do
        for mu in "${mus[@]}"; do
            for beta in "${betas[@]}"; do
                for sample_fraction in "${sample_fractions[@]}"; do
                    for algo in "${algos[@]}"; do
                        echo "algo=${algo}, sample_fractionrithm=${sample_fraction}, beta=${beta} 으로 실행 중"
                        python3 main.py --device cuda:0 --datadir ${dir} --alpha_reversed ${alpha} --alg ${algo} --lr ${lr} --mu ${mu} --sample_fraction ${sample_fraction} --beta ${beta} --batch-size 50 --comm_round 50 --n_parties 100 --dataset ${data} --model resnet18 --epochs 5 --partition noniid --reg 0.001 --save_model 1&
                        current_jobs=$((current_jobs + 1))

                        if [ "$current_jobs" -ge "$max_jobs" ]; then
                            wait -n
                            current_jobs=$((current_jobs - 1))
                        fi
                    done
                done
            done
        done
    done
done


wait
echo "모든 작업이 완료되었습니다."