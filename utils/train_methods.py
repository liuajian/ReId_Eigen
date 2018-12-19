"""
method: softmax(s)  eigen_softmax(es)  softmax_verification(sv) center_loss(cs)
Author: AjianLiu
Date: 2018.12.18
"""
from __future__ import print_function
import argparse,math,stat,subprocess
from model_libs import *
import numpy as np
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, help='s, es, sv, cs', default='s')
    parser.add_argument('--root_path', type=str, default='/home/ajliu/LAJ/ReId_Eigen')
    parser.add_argument('--caffe_root', type=str, default='/home/ajliu/LAJ/ReId_Eigen/caffe-DDM')
    parser.add_argument('--network', type=str, default='res50')
    parser.add_argument("--gpus", type=str, default="2", help="Switch for gpu computation.")
    parser.add_argument("--batch_size", type=int, default=28, help="min-batch.")
    parser.add_argument("--id_num", type=int, default=751)
    parser.add_argument("--loss_wight_s", help = 'softmaxwithloss', type=float, default=1)
    parser.add_argument("--loss_wight_e", help = 'eigen', type=float, default=0.0001)
    parser.add_argument("--loss_wight_v", help = 'verification', type=float, default=0.0001)
    parser.add_argument("--loss_wight_c", help = 'center', type=float, default=0.0001)
    parser.add_argument("--images_dim", default='224,224', help="Canonical 'height,width' dimensions of input images.")
    parser.add_argument("--mean_value", default="104,117,124",help="subtraction mean")
    parser.add_argument("--crop_size", type=int,default=224)
    parser.add_argument("--creat_prototxt", type=int, default=1)
    return parser.parse_args(argv)

def main(args):
    # We assume you are running the script at the root_path.
    method = args.method
    root_path = args.root_path
    caffe_root = args.caffe_root

    image_dims = [int(s) for s in args.images_dim.split(',')]
    mean_value = None
    if args.mean_value:
        mean_value = [float(s) for s in args.mean_value.split(',')]
        mean_value = np.array(mean_value)

    # Defining which GPUs to use.
    gpulist = args.gpus.split(",")
    num_gpus = len(gpulist)

    # Divide the mini-batch to different GPUs.
    batch_size = args.batch_size
    batch_size_per_device = batch_size
    if num_gpus > 0:
      batch_size_per_device = int(math.ceil(float(batch_size) / num_gpus))

    # Data source txt or lmdb
    train_data_lmdb = '{}/datasets/train_lmdb'.format(root_path)
    if method == 'es':train_data_txt = '{}/datasets/train_pair.txt'.format(root_path)
    else:train_data_txt = '{}/datasets/Market-1501-v15.09.15/train.lst'.format(root_path)

    # Solver parameters.
    solver_param = {
        # Train parameters
        'average_loss': 25,
        'display': 100,
        'base_lr': 0.001,
        'lr_policy': "multistep",
        'gamma': 0.1,
        'stepvalue': [20000, 50000],
        'max_iter': 50000,
        'snapshot': 1000,
        'weight_decay': 0.0005,
        'momentum': 0.9,
        'type': "SGD",
        'solver_mode': P.Solver.GPU,
    }

    # train_transform_param = {
    #     'mirror': True,
    #     'mean_file': root_path + '/datasets/' + args.images_dim[0] + '_' + args.images_dim[1]
    #                  + '_' + batch_size/2 + '_mean.binaryproto'
    # }
    train_transform_param = {
        'mirror': True,
        'crop_size': args.crop_size,
        'mean_value': [mean_value[0],mean_value[1],mean_value[2]],
        # Specify the range of scaling factor for doing resizing
        'min_scaling_factor': 0.75,
        'max_scaling_factor': 1.50,
        # Specify the angle for doing rotation
        'max_rotation_angle': 30,
        # Specify the contrast, brightness, smooth and color shift for augmentation
        'contrast_brightness_adjustment': True,
        'smooth_filtering': True,
        'min_contrast': 0.5,
        'max_contrast': 1.5,
        'max_brightness_shift': 10,
        'max_smooth': 6,
        'max_color_shift': 20,
        # Min side resizing, keep aspect ratio
        'min_side_min': 112,
        'min_side_max': 0, #168
        'min_side': 0,
        'apply_probability': 0.5,
        'debug_params': False
    }

    # Network.
    network = args.network
    # The name of the model which stores the model .prototxt file.. Modify it if you want.
    model_name = "{}/models/{}_{}".format(root_path,network,method)
    # Directory which stores the snapshot of models.
    snapshot_dir = "{}/snapshot/".format(model_name)
    # Directory which stores the log of models.
    log_dir = "{}/log".format(model_name)

    # model definition files.
    train_net_file = "{}/train.prototxt".format(model_name)
    deploy_net_file = "{}/deploy.prototxt".format(model_name)
    solver_file = "{}/solver.prototxt".format(model_name)
    # script path.
    sh_file = "{}/train.sh".format(model_name)

    # Check file.
    check_if_exist(train_data_lmdb)
    check_if_exist(train_data_txt)

    make_if_not_exist(model_name)
    make_if_not_exist(snapshot_dir)
    make_if_not_exist(log_dir)

    if args.creat_prototxt:
        if method == 'es':is_shuffle = False
        else:is_shuffle = True
        # Create train net.
        net = caffe.NetSpec()
        # net.data, net.label = CreateDataLayer_LMDB(train_data_lmdb, batch_size=batch_size_per_device,
        #                                       train=True, transform_param=train_transform_param)
        if method == 'sv':
            net.data, net.label = CreateDataLayer_SV_IMG(train_data_txt,new_width=image_dims[0],new_height=image_dims[1],
                                                         batch_size=batch_size_per_device/2,
                                                         pos_fraction=1, neg_fraction=1, pos_limit=1.0,
                                                         neg_limit=4.0, pos_factor=1.0, neg_factor=1.01,
                                                         train=True, transform_param=train_transform_param)
        else:
            net.data, net.label = CreateDataLayer_IMG(train_data_txt,shuffle=is_shuffle,new_width=image_dims[0],
                                                      new_height=image_dims[1], batch_size=batch_size_per_device,
                                                      train=True, transform_param=train_transform_param)
        if '50' in network:
            net, final = res50_body(net, 'data', '', True)

        if method == 'es':
            net['{}'.format('orig_data')],\
            net['{}'.format('eigen_data')] = L.Slice(net.pool5, ntop=2, slice_point=[batch_size_per_device/2], axis=0)
            net.euclidean_weight_loss = L.EuclideanWeightLoss(net.orig_data,net.eigen_data,net.label,loss_weight=args.loss_wight_e)
        elif method == 'sv':
            net['{}'.format('euclidean')], \
            net['{}'.format('label_dif')]  = L.PairEuclidean(net.pool5,net.label,ntop=2)
            net['{}'.format('score_dif')] = \
                L.InnerProduct(net.euclidean, num_output=2,weight_filler={'type': "gaussian",'std': 0.01})
            net.loss_dif = L.SoftmaxWithLoss(net.score_dif,net.label_dif,
                                             loss_weight = args.loss_wight_v,propagate_down=[True, False])
        elif method == 'cs':
            net.center_loss = L.CenterLoss(net.pool5,net.label,num_output=args.id_num,
                                           center_filler={'type': "xavier"},loss_weight=args.loss_wight_e)

        net.fc2 = L.InnerProduct(net.pool5, num_output=args.id_num,weight_filler={'type': "gaussian",'std': 0.01})
        net.loss = L.SoftmaxWithLoss(net.fc2, net.label,loss_param = {'ignore_label': -1},
                                     loss_weight=args.loss_wight_s,propagate_down=[True,False])


        with open(train_net_file, 'w') as f:
            print('name: "{}_train.prototxt"'.format(network), file=f)
            print(net.to_proto(), file=f)

        # Create deploy net.
        deploy_net = net
        with open(deploy_net_file, 'w') as f:
            net_param = deploy_net.to_proto()
            del net_param.layer[0]
            if method == 'es' or method == 'sv':
                if method == 'sv':
                    del net_param.layer[-5]
                del net_param.layer[-4]
                del net_param.layer[-3]
            elif method == 'cs':
                del net_param.layer[-3]
            del net_param.layer[-2]
            del net_param.layer[-1]
            net_param.input.extend(['data'])
            net_param.input_shape.extend([caffe_pb2.BlobShape(dim=[1, 3, image_dims[0],image_dims[1]])])
            print(net_param, file=f)

        # Create solver.
        solver = caffe_pb2.SolverParameter(
                train_net=train_net_file,
                snapshot_prefix=snapshot_dir + '{}_{}'.format(network,method),
                **solver_param)
        with open(solver_file, 'w') as f:
            print(solver, file=f)

        # Create job file.
        with open(sh_file, 'w') as f:
            f.write('cd {}\n'.format(caffe_root))
            f.write('./build/tools/caffe train \\\n')
            f.write('--solver="{}" \\\n'.format(solver_file))
            f.write('--weights="{}" \\\n'.format(root_path + '/models/ResNet_50.caffemodel'))
            f.write('--gpu {} 2>&1 | tee {}/{}_{}.log\n'.format(args.gpus, log_dir, network,method))

    # Run the job.
    os.chmod(sh_file, stat.S_IRWXU)
    subprocess.call(sh_file, shell=True)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
