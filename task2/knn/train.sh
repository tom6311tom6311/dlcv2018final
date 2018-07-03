mkdir -p model
mkdir -p log
python3 code/knn_train.py --train $1 -m model -l log
