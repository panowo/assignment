
cd /project/msbd5009_stu01/ass3/src

nvcc -std=c++11 cuda_skeleton.cu decom.cpp main.cpp -o lrds

./lrds -ParallelQuery ../data/WR/ 4 128

