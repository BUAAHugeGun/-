export PYTHONPATH=..:$PYTHONPATH
CUDA_VISIBLE_DEVICES=$2 python synthesis_train.py --root $1 \
