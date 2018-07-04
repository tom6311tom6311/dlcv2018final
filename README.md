### README
####Task 1
Model 已儲存於 repo 中， testing script 如下：
- $> `cd task1/`
- $> `./test.sh <data_dir> out.csv`
--> 其中 *<data_dir>* 為包含 `test/` 資料夾的母資料夾
--> 會輸出 `out.csv`

####Task 2
我們實作了4種模型，其中以 (3) performance 最佳，因此**助教可直接跑 (3)**  
各模型 training 與 testing script如下：

(1) Siamese Network
- $> `cd task2/siamese_ping/`
- $> `./train_and_test_1.sh <train_data_dir> <test_data_dir>`
--> 會輸出 `out_1.csv`，為 1-shot 結果
- $> `./train_and_test_5.sh <train_data_dir> <test_data_dir>`
--> 會輸出 `out_5.csv`，為 5-shot 結果
- $> `./train_and_test_10.sh <train_data_dir> <test_data_dir>`
--> 會輸出 `out_10.csv`，為 10-shot 結果

(2) Relation Network
- $> `cd task2/relationnet/`
- $> `./train.sh <train_data_dir> 1`
- $> `./test.sh <train_data_dir> <test_data_dir> ./ 1`
--> 會輸出 `Relation_sample_1_predict.csv`，為 1-shot 結果
- $> `./train.sh <train_data_dir> 5`
- $> `./test.sh <train_data_dir> <test_data_dir> ./ 5`
--> 會輸出 `Relation_sample_5_predict.csv`，為 5-shot 結果
- $> `./train.sh <train_data_dir> 10`
- $> `./test.sh <train_data_dir> <test_data_dir> ./ 10`
--> 會輸出 `Relation_sample_10_predict.csv`，為 10-shot 結果

(3) Matching Network  
Model 已儲存於 repo 中， testing script 如下：
- $> `cd task2/matchingnet/`
- $> `./matching_test.sh <train_data_dir> <test_data_dir> out_1.csv out_5.csv out_10.csv`
--> 會輸出 `out_1.csv`、 `out_5.csv`、`out_10.csv`，分別為 1/5/10-shot 的結果

(4) CNN + (PCA) + KNN
- $> `cd task2/knn/`
- $> `./train.sh <train_data_dir>`
- $> `./test_1.sh <train_data_dir> <test_data_dir> ./`
--> 會輸出 `1_PCA_knn_predict.csv` 及 `1_knn_predict.csv`，分別為有/無PCA機制的 1-shot 結果
- $> `./test_5.sh <train_data_dir> <test_data_dir> ./`
--> 會輸出 `5_PCA_knn_predict.csv` 及 `5_knn_predict.csv`，分別為有/無PCA機制的 5-shot 結果
- $> `./test_10.sh <train_data_dir> <test_data_dir> ./`
--> 會輸出 `10_PCA_knn_predict.csv` 及 `10_knn_predict.csv`，分別為有/無 PCA 機制的 10-shot 結果
