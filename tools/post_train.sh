export PYTHONPATH=..:$PYTHONPATH
CUDA_VISIBLE_DEVICES=$2 python post_train.py --root $1 \
