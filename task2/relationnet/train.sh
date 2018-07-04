mkdir -p log
mkdir -p model
python3 code/train.py --train $1 -s $2 -m model -l log
