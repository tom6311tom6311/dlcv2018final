#!/bin/bash

if [ $1 ]; then
  data_dir=$1
else
  data_dir="data"
fi

if [ ! -d "$1" ]; then
  mkdir $data_dir
else
  rm -r $data_dir/*
fi

curl -o $data_dir/task2-train-dataset.tar -L https://www.dropbox.com/s/04dch6dsu6z7c39/task2-train-dataset.tar?dl=1
tar -xvf $data_dir/task2-train-dataset.tar -C $data_dir
mv $data_dir/task2-dataset/* $data_dir
rm $data_dir/task2-train-dataset.tar
rm -r $data_dir/task2-dataset
