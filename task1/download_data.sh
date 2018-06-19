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

curl -o $data_dir/Fashion_MNIST_student.tar -L https://www.dropbox.com/s/k8irvszvz8d2kbv/Fashion_MNIST_student.tar?dl=1
tar -xvf $data_dir/Fashion_MNIST_student.tar -C $data_dir
mv $data_dir/Fashion_MNIST_student/* $data_dir
rm $data_dir/Fashion_MNIST_student.tar
rm -r $data_dir/Fashion_MNIST_student
