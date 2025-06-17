#!/bin/bash

GPU_NUM=2
dataset_name=(az lj ls nf)
# dataset_name=(nf)
datasets=(amazon-dyna/6 livejournal-dyna/30 lsbench-dyna/1 netflow-dyna)
# datasets=(netflow-dyna)
# querys=(sparse_6 dense_6 tree_6)
# querys=(query_4 query_6 query_8 query_10 query_12)
querys=(query_5 query_7)

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


# dataset_id=0
# dataset=${datasets[dataset_id]}
# query=tree_6
# file=output_${dataset_name[dataset_id]}_${query}.txt
# # echo "Running on $dataset, query: $query" > $file
# for i in $(seq 8 99)
# do
#   echo "" >> $file
#   q=Q_$i.in
#   echo $q >> $file
#   timeout 600 ./build/JoMatch -q ../datasets/$dataset/query_graph/$query/$q -d ../datasets/$dataset/data_graph/data.dyna --gpu $GPU_NUM >> $file
# done

# dataset_id=0
for dataset_id in $(seq 1 3)
do
  dataset=${datasets[dataset_id]}
  for query in ${querys[@]}
  do
    # echo "Running on dataset: $dataset" > output.txt
    file=output_${dataset_name[dataset_id]}_${query}.txt
    echo "Running on $dataset, query: $query" > $file
    num_qs=$(ls ../datasets/$dataset/$query | wc -l)
    # for i in $(seq 0 $((num_qs-1)))
    for i in $(seq 0 9)
    # for i in $(seq 0 99)
    do
      echo "" >> $file
      q=Q_$i.in
      echo $q >> $file
      timeout 900 ./build/JoMatch -q ../datasets/$dataset/$query/$q -d ../datasets/$dataset/data_graph/data.dyna --gpu $GPU_NUM >> $file
    done
  done
done