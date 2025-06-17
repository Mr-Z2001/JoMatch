#!/bin/bash

GPU_NUM=0
dataset_name=(az lj ls nf)
# dataset_name=(lj ls nf)
datasets=(amazon-dyna/6 livejournal-dyna/30 lsbench-dyna/1 netflow-dyna)
# datasets=(amazon-dyna/6)
# rates=("1" "3" "7" "9")
# datasets=(livejournal-dyna/30 lsbench-dyna/1 netflow-dyna)
# subquerys=(query_10 query_12)
# querys=(query_4 query_6 query_8 query_10 query_12)
querys=(query_6)

# dataset=amazon-dyna/6
# compute-sanitizer ./build/JoMatch -q ../datasets/$dataset/query_graph/tree_6/Q_1.in -d ../datasets/$dataset/data_graph/data.dyna --gpu $GPU_NUM > output_tree.txt

# file=output_lj_dense_6.txt
# for i in $(seq 0 99)
# do
#   echo "" >> $file
#   q=Q_$i.in
#   echo $q >> $file
#   ./build/JoMatch -q ../datasets/livejournal-dyna/30/query_graph/dense_6/$q -d ../datasets/livejournal-dyna/30/data_graph/data.dyna --gpu $GPU_NUM >> $file
# done

# file=output_lj_tree_6.txt
# for i in $(seq 0 99)
# do
#   echo "" >> $file
#   q=Q_$i.in
#   echo $q >> $file
#   ./build/JoMatch -q ../datasets/livejournal-dyna/30/query_graph/tree_6/$q -d ../datasets/livejournal-dyna/30/data_graph/data.dyna --gpu $GPU_NUM >> $file
# done

# dataset_id=1
# dataset=${datasets[dataset_id]}
# query=sparse_6
# file=output_${dataset_name[dataset_id]}_${query}.txt
# echo "Running on $dataset, query: $query" > $file
# for i in $(seq 0 99)
# do
#   echo "" >> $file
#   q=Q_$i.in
#   echo $q >> $file
#   timeout 600 ./build/JoMatch -q ../datasets/$dataset/query_graph/$query/$q -d ../datasets/$dataset/data_graph/data.dyna --gpu $GPU_NUM >> $file
# done

# dataset_id=0
# dataset=${datasets[dataset_id]}
# for query in ${subquerys[@]}
# do
#   # echo "Running on dataset: $dataset" > output.txt
#   file=output_${dataset_name[dataset_id]}_${query}.txt
#   echo "Running on $dataset, query: $query" > $file
#   num_qs=$(ls ../datasets/$dataset/$query | wc -l)
#   for i in $(seq 0 $((num_qs-1)))
#   # for i in $(seq 0 99)
#   do
#     echo "" >> $file
#     q=Q_$i.in
#     echo $q >> $file
#     timeout 900 ./build/JoMatch -q ../datasets/$dataset/$query/$q -d ../datasets/$dataset/data_graph/data.dyna --gpu $GPU_NUM >> $file
#   done
# done


# for dataset_id in $(seq 1 3)
# do
#   dataset=${datasets[dataset_id]}
#   for query in ${querys[@]}
#   do
#     # echo "Running on dataset: $dataset" > output.txt
#     file=output_${dataset_name[dataset_id]}_${query}.txt
#     echo "Running on $dataset, query: $query" > $file
#     num_qs=$(ls ../datasets/$dataset/$query | wc -l)
#     for i in $(seq 0 $((num_qs-1)))
#     # for i in $(seq 0 99)
#     do
#       echo "" >> $file
#       q=Q_$i.in
#       echo $q >> $file
#       timeout 900 ./build/JoMatch -q ../datasets/$dataset/$query/$q -d ../datasets/$dataset/data_graph/data.dyna --gpu $GPU_NUM >> $file
#     done
#   done
# done

# for rate in ${rates[@]}
# do
#   file=output_${rate}.txt
#   echo "Running on rate: $rate" > $file
#   for q in $(seq 0 99)
#   do
#     echo "" >> $file
#     echo "Q_$q.in" >> $file
#     timeout 900 ./build/JoMatch -q ../datasets/amazon-dyna/6/query_graph/sparse_6/Q_$q.in -d ../datasets/amazon-dyna/6/data_graph/data_${rate}.dyna --gpu $GPU_NUM >> $file
#   done  
# done

for dataset_id in $(seq 1 2)
do
  dataset=${datasets[dataset_id]}
  file=output_${dataset_name[dataset_id]}_pat.txt
  for pat_id in $(seq 0 32)
  do
    # echo "Running on dataset: $dataset" > output.txt
    echo "Pat: $pat_id" >> $file
    timeout 900 ./build/JoMatch -q ../datasets/amazon/pat/Q_$pat_id -d ../datasets/$dataset/data_graph/data.dyna --gpu $GPU_NUM >> $file
  done
done

# for pat_id in $(seq 0 32)
# do
#   file=output_pat_${pat_id}.txt
#   echo "Running on pattern id: $pat_id" > $file
#   for q in $(seq 0 99)
#   do
#     echo "" >> $file
#     echo "Q_$q.in" >> $file
#     timeout 900 ./build/JoMatch -q ../datasets/amazon-dyna/6/query_graph/sparse_6/Q_$q.in -d ../datasets/amazon-dyna/6/data_graph/data_pat_${pat_id}.dyna --gpu $GPU_NUM >> $file
#   done
# done