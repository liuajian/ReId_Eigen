"""
Extracting features
Author: AjianLiu
Date: 2018.12.18
"""
from model_libs import *
import argparse
import time,sys,random,cv2
import numpy as np
import scipy.io as sio

def main(argv):
  parser = argparse.ArgumentParser()
  # Required arguments: input and output files.
  parser.add_argument('--network', type=str, help='res50,res18', default='res50')
  parser.add_argument('--method', type=str, help='s, es, sv, cs', default='s')
  parser.add_argument("--Query_file", default='query.lst', help="Probe")
  parser.add_argument("--Test_file", default='test.lst', help="Gallery")
  parser.add_argument("--out_features", default='/home/ajliu/LAJ/ReId_Eigen/out_features/', help="Feature storage path")
  # Optional arguments.
  parser.add_argument("--batch_size", type=int,default=14)
  parser.add_argument("--model_def",help="deploy.ptotxt")
  parser.add_argument("--pretrained_model",help="caffe.model")
  parser.add_argument("--gpu",type=int,default=-1,help="Switch for gpu computation.")
  parser.add_argument("--images_dim",default='224,224',help="Canonical 'height,width' dimensions of input images.")
  parser.add_argument("--mean_value",help="subtraction mean")
  parser.add_argument("--feature_name",default="pool5",help="feature blob name.")
  args = parser.parse_args()
  #############
  method = args.method
  network = args.network
  Txt_list = [args.Query_file, args.Test_file]
  Mat_list = [args.out_features + network + '_' + method + '_python_' + os.path.basename(args.Query_file)+'.feature.mat',
              args.out_features + network + '_' + method + '_python_' + os.path.basename(args.Test_file) +'.feature.mat']
  if not os.path.exists(args.out_features):
    os.makedirs(args.out_features)
  if args.gpu >= 0:
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu)
    print("GPU mode, device : {}".format(args.gpu))
  else:
    caffe.set_mode_cpu()
    print("CPU mode")
  image_dims = [int(s) for s in args.images_dim.split(',')]
  mean_value = None
  if args.mean_value:
    mean_value = [float(s) for s in args.mean_value.split(',')]
    mean_value = np.array(mean_value)
  start_time = time.time()
 ### starting ###
  net = caffe.Net(args.model_def, args.pretrained_model, caffe.TEST)
  for i in range(0,len(Txt_list)):
    img_files, labels = load_txt(Txt_list[i])
    print("Loading file: %s, Number images: %d" % (Txt_list[i], len(img_files)))
    if i == 0: ### Query
      query_files = img_files
      fea = \
        extracting_feature_batch_size(net, query_files, args.batch_size, args.feature_name, image_dims, mean_value)
    else:      ### Test
      test_files = img_files
      fea = \
        extracting_feature_batch_size(net, test_files, args.batch_size, args.feature_name, image_dims, mean_value)
    #### Save
    sio.savemat(Mat_list[i], {'feature': fea})
  ### END ###
  end_time = time.time() - start_time
  need_hour, need_mins, need_secs = convert_secs2time(end_time)
  need_time = '{:02d}:{:02d}:{:02d}'.format(need_hour, need_mins, need_secs)
  print("need {:s}.".format(need_time))


### utils ###
def shuf_list(list):
    random.shuffle(list)

def load_txt(xfile):
  img_files = []
  labels = []
  for line in open(xfile):
    line = line.strip('\n').split(' ')
    assert(len(line) == 2)
    img_files.append(line[0])
    labels.append(int(float(line[1])))
  return img_files, labels

def convert_secs2time(epoch_time):
    need_hour = int(epoch_time / 3600)
    need_mins = int((epoch_time - 3600*need_hour) / 60)
    need_secs = int(epoch_time - 3600*need_hour - 60*need_mins)
    return need_hour, need_mins, need_secs

### extracting features by caffe ###
def extracting_feature_batch_size(net,test_path,batch_size,feature_name,image_dims,mean_value):
    fea = []
    paths_all = test_path
    ls_len = len(paths_all)
    div_lines = []
    for i in range(0, ls_len, batch_size):
      div_lines.append(paths_all[i:i + batch_size])
    for it in range(len(div_lines)):
      num_image = len(div_lines[it])
      blob = np.zeros((num_image, image_dims[0], image_dims[1], 3), dtype=np.float32)
      for num in range(num_image):
        line = div_lines[it][num]
        im = cv2.imread(line).astype(np.float32)
        im = cv2.resize(im, (image_dims[0], image_dims[1]))
        im -= mean_value
        im = np.expand_dims(im, axis=0)
        blob[num, 0:image_dims[0], 0:image_dims[1], :] = im
      blob = blob.transpose((0, 3, 1, 2)) #
      net.blobs['data'].reshape(num_image, 3, image_dims[0], image_dims[1])
      net.blobs['data'].data[...] = blob
      # 1
      # feature = net.forward()[feature_name]
      # 2
      net.forward()
      feature = net.blobs[feature_name]
      for i in range(0,num_image):
        fea.append(np.copy(feature.data[i]))
      print('feature : {} : {}/{} : {} : {}'
            .format(feature_name, it + 1,len(div_lines),np.array(fea).shape, np.mean(fea)))
    return fea


if __name__ == '__main__':
  main(sys.argv)


