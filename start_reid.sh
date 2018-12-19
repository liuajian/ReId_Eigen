#!/usr/bin/env bash
# We assume you are running the script at the $root_path of ReId_Eigen(~).
DATE=$(date +%F-%H:%M:%S)
echo "Time:" "$DATE" "Date: 2018.12.18 Author: AjianLiu Email: ajianliu92@gmail.com"> results.txt
Step_1=true
Step_2=true
Step_3=true
Step_4=true
###########
### Download the Market-1501-v15.09.15.zip and place it in the $datasets directory
###########
root_path=$(pwd)
data_name=market1501
Id_num=751
network=res50
gpu_id=1
Batch_size=14
max_iter=50000
feature_name="pool5"
Mean_value="104,117,124"
data_lmdb_path=$root_path/datasets
eigenbody_data=$root_path/datasets/eigen_v2
data_path=$root_path/datasets/Market-1501-v15.09.15
TOOLS=$root_path/caffe-DDM/build/tools
fea_path=$root_path/out_features
if [ ${network} == "res50" ]; then
   Imae_h=224
   Imae_w=224
   Crop_size=224
else
   Imae_h=112
   Imae_w=112
   Crop_size=112
fi
###########
if $Step_1; then
echo "Step 1: Generating the train.txt, query.txt, test.txt, training_pair.txt ..."
cd datasets
unzip Market-1501-v15.09.15.zip
unzip eigen_v2.zip
cd ..
cd evaluation
matlab -nodesktop -nosplash -r \
          "clc;clear all; \
           root_dir='$data_path';generate_train";
cd ..
Is_shuffle=true
python utils/generate_txt.py \
          $data_path/ \
          $eigenbody_data/ \
          --batch_size 14 \
          --is_shuffle $Is_shuffle
fi

if $Step_2; then
echo "Step 2: Convert images to lmdb"
train_name=train_lmdb
test_name=test_lmdb
RESIZE=true
if $RESIZE; then
  RESIZE_HEIGHT=$Imae_h
  RESIZE_WIDTH=$Imae_w
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi
echo "Create train lmdb.."
rm -rf $data_lmdb_path/$train_name
GLOG_logtostderr=1 $TOOLS/convert_imageset \
           --shuffle=false \
           --resize_height=$RESIZE_HEIGHT \
           --resize_width=$RESIZE_WIDTH \
           / \
           $data_lmdb_path/train_pair.txt \
           $data_lmdb_path/$train_name
echo "Creat mean proto"
GLOG_logtostderr=1 $TOOLS/compute_image_mean $data_lmdb_path/$train_name \
$data_lmdb_path/${Imae_h}_${Imae_h}_${Batch_size}_mean.binaryproto
echo Generate ${Imae_h}_${Imae_h}_$[Batch_size*2]_mean.binaryproto
fi

echo "methods: softmax(s)  eigen_softmax(es)  softmax_verification(sv) center_loss(cs)"
methods=([0]="s" [1]="es" [2]="sv" [3]="cs")
if $Step_3; then
echo "Step 3: Training"
  for((i=0;i<=3;i++));
  do
    python utils/train_methods.py \
          --method ${methods[$i]} \
          --root_path $root_path \
          --caffe_root $root_path/caffe-DDM \
          --network $network \
          --gpus $gpu_id \
          --batch_size $Batch_size \
          --id_num $Id_num \
          --loss_wight_s 1 \
      	  --loss_wight_e 0.0001 \
          --loss_wight_v 0.5 \
          --loss_wight_c 0.0001 \
          --images_dim $Imae_h,$Imae_w \
          --mean_value $Mean_value \
          --crop_size $Crop_size \
          --creat_prototxt 1

  done
fi

if $Step_4; then
  echo "Step 4: Features extracted by python"
  txt_name=([0]="query.lst" [1]="test.lst")
  for((i=0;i<=3;i++));
  do
    model_file=$root_path/models/${network}_${methods[$i]}/snapshot/${network}_${methods[$i]}_iter_${max_iter}.caffemodel
    echo "$model_file"
    python utils/extract_feature.py \
      --network $network \
      --method ${methods[$i]} \
	  --Query_file $data_path/${txt_name[0]} \
	  --Test_file $data_path/${txt_name[1]} \
	  --out_features $fea_path/ \
	  --batch_size $Batch_size \
	  --model_def $root_path/models/${network}_${methods[$i]}/deploy.prototxt \
          --pretrained_model $model_file \
	  --gpu $gpu_id \
	  --images_dim $Imae_h,$Imae_w \
	  --mean_value 104,117,124 \
	  --feature_name $feature_name

    echo "Step5: Testing the features extracted by python with matlab"
    cd evaluation
    matlab -nodesktop -nosplash -r \
          "clc;clear all; \
           netname='${network}_${methods[$i]}'; \
           root_path='$root_path'; \
           fea_mat_path='$fea_path'; \
           is_extracting=false; \
           type_fea='python';Market1501";
    cd ..
  done
fi


