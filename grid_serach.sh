#!/bin/sh
declare -a n_estimators_values=(400 500 600)
declare -a max_depth_values=(10 20 30)
declare -a na_strategies=('zero')
declare -a mag_max_target_th=(5 6)



for n_est in "${n_estimators_values[@]}"
    do
        for max_d in "${max_depth_values[@]}"
            do
                for na_s in "${na_strategies[@]}"
                    do
                        for target_th in "${mag_max_target_th[@]}"
                            do
                            dvc exp run --queue -S random_forest.params.n_estimators=$n_est -S random_forest.params.max_depth=$max_d -S target.mag_max_target_th=$target_th -S preprocess.na_strategy=$na_s --rev lightgbm
                            done
                    done
            done
    done