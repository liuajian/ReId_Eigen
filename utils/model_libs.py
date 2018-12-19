import os,sys
sys.path.insert(0, './caffe-DDM/python')
import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2

### define data layer ###
def CreateDataLayer_LMDB(source, batch_size=28, backend=P.Data.LMDB, train=True, transform_param={}):
    if train:
        kwargs = {
                'include': dict(phase=caffe_pb2.Phase.Value('TRAIN')),
                'transform_param': transform_param,
                }
    else:
        kwargs = {
                'include': dict(phase=caffe_pb2.Phase.Value('TEST')),
                'transform_param': transform_param,
                }
    return L.Data(name="data", data_param=dict(batch_size=batch_size,
                                               backend=backend,
                                               source=source),
                                               ntop=2, **kwargs)

def CreateDataLayer_IMG(source, batch_size=28,shuffle=True,
                        new_width=224,new_height=224,is_color=True,train=True, transform_param={}):
    if train:
        kwargs = {
                'include': dict(phase=caffe_pb2.Phase.Value('TRAIN')),
                'transform_param': transform_param,
                }
    else:
        kwargs = {
                'include': dict(phase=caffe_pb2.Phase.Value('TEST')),
                'transform_param': transform_param,
                }
    return L.ImageData(name="data",image_data_param=dict(source=source,
                                                         shuffle=shuffle,
                                                         new_width=new_width,new_height=new_height,
                                                         is_color=is_color,
                                                         batch_size=batch_size),
                                                         ntop=2, **kwargs)

def CreateDataLayer_SV_IMG(source, batch_size=28,new_width=224,new_height=224,
                           pos_fraction=1,neg_fraction=1,pos_limit=1.0,neg_limit=4.0,pos_factor=1.0,neg_factor=1.01,
                           is_color=True,train=True, transform_param={}):
    if train:
        kwargs = {
                'include': dict(phase=caffe_pb2.Phase.Value('TRAIN')),
                'transform_param': transform_param,
                }
    else:
        kwargs = {
                'include': dict(phase=caffe_pb2.Phase.Value('TEST')),
                'transform_param': transform_param,
                }
    return L.ReidData(name="data",reid_data_param=dict(source=source,
                                                         new_width=new_width,new_height=new_height,
                                                         pos_fraction=1, neg_fraction=1, pos_limit=1.0,
                                                         neg_limit=4.0,pos_factor=1.0, neg_factor=1.01,
                                                         is_color=is_color,
                                                         batch_size=batch_size),
                                                         ntop=2, **kwargs)

### define network ###
def check_if_exist(path):
  return os.path.exists(path)

def make_if_not_exist(path):
  if not os.path.exists(path):
    os.makedirs(path)

# helper function for common structures
def conv_relu(bottom, ks, nout, stride=1, pad=0, group=1, param=None, is_train=False, has_relu=True, bias_term=True, base_param_name=None):
  kwargs = {'num_output': nout, 'kernel_size': ks} 
  if (stride != 1): 
    kwargs['stride'] = stride
  if (group != 1): 
    kwargs['group'] = group
  if (pad != 0): 
    kwargs['pad'] = pad 
  if (bias_term==False):
    kwargs['bias_term'] = bias_term
    if param != None and len(param) == 2:
      param = param[0]
  if (is_train):
    #kwargs['weight_filler'] = dict(type='gaussian', std=0.01)
    kwargs['weight_filler'] = dict(type='xavier')
    #kwargs['bias_filler'] = dict(type='constant', value=1)
  else:
    param = None
    base_param_name=None

  if (param!= None):
    #kwargs['param'] = [dict(name=param_name)]
    kwargs['param'] = param
  elif (base_param_name):
    if (bias_term):
      kwargs['param'] = [dict(name=base_param_name+'_conv_w'), dict(name=base_param_name+'_conv_b')]
    else:
      kwargs['param'] = [dict(name=base_param_name+'_conv_w')]

  conv = L.Convolution(bottom, **kwargs)
  if (has_relu):
    return conv, L.ReLU(conv, in_place=True)
  else:
    return conv


def conv_bn_scale(bottom, ks, nout, stride=1, pad=0, group=1, param=None, is_train=False, bn_param=None, scale_param=None, has_relu=False, base_param_name=None, bias_term=True):
  conv = conv_relu(bottom, ks, nout, stride=stride, pad=pad, group=group, param=param, is_train=is_train, has_relu=False, base_param_name=base_param_name, bias_term=bias_term)
  if bn_param:
    bn = L.BatchNorm(conv, use_global_stats=True, in_place=True, param=bn_param)
  elif base_param_name and is_train==True:
    bn = L.BatchNorm(conv, use_global_stats=True, in_place=True, param=[dict(name=base_param_name+'_bn1',lr_mult=0,decay_mult=0), dict(name=base_param_name+'_bn2',lr_mult=0,decay_mult=0), dict(name=base_param_name+'_bn3',lr_mult=0,decay_mult=0)])
  else:
    bn = L.BatchNorm(conv, use_global_stats=True, in_place=True)

  if scale_param:
    scale = L.Scale(bn, bias_term=True, in_place=True, param=scale_param)
  elif base_param_name and is_train==True:
    scale = L.Scale(bn, bias_term=True, in_place=True, param=[dict(name=base_param_name+'_scale1'), dict(name=base_param_name+'_scale2')])
  else:
    scale = L.Scale(bn, bias_term=True, in_place=True)

  if (has_relu):
    return conv, bn, scale, L.ReLU(scale, in_place=True)
  else:
    return conv, bn, scale

# yet another helper function
def max_pool(bottom, ks, stride=1):
  if (stride != 1):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)
  else:
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks)

def ave_pool(bottom, ks, stride=1):
  if (stride != 1):
    return L.Pooling(bottom, pool=P.Pooling.AVE, kernel_size=ks, stride=stride)
  else:
    return L.Pooling(bottom, pool=P.Pooling.AVE, kernel_size=ks)

def fc_relu(bottom, nout, param=None, is_train=False, has_relu=True, base_param_name=None):
  kwargs = {'num_output': nout}
  if (is_train):
    kwargs['weight_filler'] = dict(type='gaussian', std=0.01)
    #kwargs['weight_filler'] = dict(type='xavier')
  else:
    param = None
  if (param != None):
    #kwargs['param'] = [dict(name=param_name)]
    kwargs['param'] = param
  elif (base_param_name): 
    kwargs['param'] = [dict(name=base_param_name+'_fc_w'), dict(name=base_param_name+'_fc_b')]
  
  fc = L.InnerProduct(bottom, **kwargs)
  if (has_relu):
    return fc, L.ReLU(fc, in_place=True)
  else:
    return fc

### resnet50 ###
def res_unit(net, bottom, in_c, out_c, stride, base_name, post, is_train=False):
  assert (out_c % 4 == 0)
  # param = [dict(lr_mult=0.1,decay_mult=1), dict(lr_mult=0.2,decay_mult=0)]
  param = None

  pase_name = base_name
  base_name = base_name + post
  if (in_c != out_c):
    # param = pase_name + '_branch1'
    net['res' + base_name + '_branch1'], net['bn' + base_name + '_branch1'], net['scale' + base_name + '_branch1'] = \
      conv_bn_scale(bottom, 1, out_c, stride=stride, is_train=is_train, has_relu=False, bias_term=False, param=param)
    # conv_bn_scale(bottom,  1, out_c, base_param_name=param, stride=stride, is_train=is_train, has_relu=False, bias_term=False)
    identity = net['scale' + base_name + '_branch1']
  else:
    identity = bottom

  # param = pase_name + '_branch2a'
  net['res' + base_name + '_branch2a'], net['bn' + base_name + '_branch2a'], net['scale' + base_name + '_branch2a'], \
  net['res' + base_name + '_branch2a_relu'] = \
    conv_bn_scale(bottom, 1, out_c / 4, stride=stride, is_train=is_train, has_relu=True, bias_term=False, param=param)
  # conv_bn_scale(bottom,  1, out_c/4, base_param_name=param, stride=stride, is_train=is_train, has_relu=True, bias_term=False)

  # param = pase_name + '_branch2b'
  net['res' + base_name + '_branch2b'], net['bn' + base_name + '_branch2b'], net['scale' + base_name + '_branch2b'], \
  net['res' + base_name + '_branch2b_relu'] = \
    conv_bn_scale(net['res' + base_name + '_branch2a_relu'], 3, out_c / 4, pad=1, is_train=is_train, has_relu=True,
                  bias_term=False, param=param)
  # conv_bn_scale(net['res'+base_name+'_branch2a_relu'],  3, out_c/4, base_param_name=param, pad=1, is_train=is_train, has_relu=True, bias_term=False)

  # param = pase_name + '_branch2c'
  net['res' + base_name + '_branch2c'], net['bn' + base_name + '_branch2c'], net['scale' + base_name + '_branch2c'] = \
    conv_bn_scale(net['res' + base_name + '_branch2b_relu'], 1, out_c, is_train=is_train, has_relu=False,
                  bias_term=False, param=param)
  # conv_bn_scale(net['res'+base_name+'_branch2b_relu'],  1, out_c, base_param_name=param, is_train=is_train, has_relu=False, bias_term=False)

  final = net['scale' + base_name + '_branch2c']

  net['res' + base_name] = L.Eltwise(identity, final)
  net['res' + base_name + '_relu'] = L.ReLU(net['res' + base_name], in_place=True)
  final_name = 'res' + base_name + '_relu'
  return net, final_name


def res50_body(net, data, post, is_train):
  net['conv1' + post], net['bn_conv1' + post], net['scale_conv1' + post], net['conv1_relu' + post] = \
    conv_bn_scale(net[data], 7, 64, pad=3, stride=2, is_train=is_train, has_relu=True)

  net['pool1' + post] = max_pool(net['conv1_relu' + post], 3, stride=2)
  names, outs = ['2a', '2b', '2c'], [256, 256, 256]
  pre_out = 64
  final = 'pool1' + post
  for (name, out) in zip(names, outs):
    net, final = res_unit(net, net[final], pre_out, out, 1, name, post, is_train=is_train)
    pre_out = out

  names, outs = ['3a', '3b', '3c', '3d'], [512, 512, 512, 512]
  for (name, out) in zip(names, outs):
    if (name == '3a'):
      net, final = res_unit(net, net[final], pre_out, out, 2, name, post, is_train=is_train)
    else:
      net, final = res_unit(net, net[final], pre_out, out, 1, name, post, is_train=is_train)
    pre_out = out

  names = ['4a', '4b', '4c', '4d', '4e', '4f']
  out = 1024
  for name in names:
    if (name == '4a'):
      net, final = res_unit(net, net[final], pre_out, out, 2, name, post, is_train=is_train)
    else:
      net, final = res_unit(net, net[final], pre_out, out, 1, name, post, is_train=is_train)
    pre_out = out

  names = ['5a', '5b', '5c']
  out = 2048
  for name in names:
    if (name == '5a'):
      net, final = res_unit(net, net[final], pre_out, out, 2, name, post, is_train=is_train)
    else:
      net, final = res_unit(net, net[final], pre_out, out, 1, name, post, is_train=is_train)
    pre_out = out

  net['pool5' + post] = ave_pool(net[final], 7, 1)
  final = 'pool5' + post

  return net, final