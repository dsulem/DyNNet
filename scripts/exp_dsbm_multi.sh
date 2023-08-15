
python scripts/synthetic_multiple_cps.py --q 0.02 --p 0.08 --sizes_clique 50 60 --tolerance 3 --window_lengths 6 \
 --rep 10 --n_pairs 9000 --n_samples_train 3000 --n_samples_test 100 --n_change_points 4 --patience 30 --lr 0.01 --dropout 0.05 \
 --n_workers 1 --nepochs 100 --cuda 3


python scripts/synthetic_multiple_cps.py --q 0.02 --p 0.08 --sizes_clique 50 --tolerance 3 --window_lengths 6 \
--rep 10 --n_pairs 9000 --n_samples_train 3000 --n_samples_test 100 --n_change_points 4 --patience 30 --lr 0.01 --dropout 0.05 \
--n_workers 1 --nepochs 100 --top_k 30 50 100 --cuda 2