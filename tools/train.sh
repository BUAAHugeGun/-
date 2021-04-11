export PYTHONPATH=..:$PYTHONPATH
CUDA_VISIBLE_DEVICES=$2 python train.py --root $1 \
