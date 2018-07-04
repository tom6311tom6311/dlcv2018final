#!/bin/bash
python3 matching_test.py $1 $2 model/matching1_finetune2.h5 $3 1
python3 matching_test.py $1 $2 model/matching5_finetune2.h5 $4 5
python3 matching_test.py $1 $2 model/matching10_finetune2.h5 $5 10
