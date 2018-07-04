# train
./matching_train.sh [training_data_path] [model_path] [n shot]
Ex: ./matching_train.sh data/task2-dataset model/matching1.h5 1

# test
./matching_test.sh [training_data_path] [test_data_path] [1 shot outputfile] [5 shot outputfile] [10 shot outputfile]
Ex: ./matching_test.sh data/task2-dataset data/test 1.csv 5.csv 10.csv