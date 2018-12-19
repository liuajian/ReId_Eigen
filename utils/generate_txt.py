"""
Generate a data paired txt file for caffe training
"""
import sys,argparse
import random,os
### the following does not need to be modified #####
def main(args):
    data_list = 'train.lst'
    with open(args.input_dir + data_list) as f:
        lines = f.readlines()
    train_txt = []
    if args.is_shuffle:
        random.shuffle(lines)
    ls_len = len(lines)
    div_lines = []
    for i in range(0, ls_len, args.batch_size):
        div_lines.append(lines[i:i+args.batch_size])
    for it in range(len(div_lines)):   # it means the number of batch_size
        num_image = len(div_lines[it]) # num_image = batch_size
        if num_image != args.batch_size:
            continue
        for num in range(num_image):
            index_a = 2*it* args.batch_size +num
            index_b = index_a + num_image
            line = div_lines[it][num]
            line = line.strip().split(' ')
            train_txt.insert(index_a, line[0]+' '+line[1])
            # eigen_line = args.eigen_dir + line[1] + '.jpg' +' '+line[1]
            eigen_line = args.eigen_dir + line[1] + '.jpg' + ' -1'
            train_txt.insert(index_b, eigen_line)
    print('input_data = %d, batch_size = %d, num_batch = %d, is_shuffle = %s' % (
    len(lines), args.batch_size, len(div_lines), args.is_shuffle))
    print('checking: %d * %d + %d =? %d' % (
    args.batch_size, len(div_lines) - 1, len(div_lines[len(div_lines) - 1]), len(lines)))
    if len(train_txt)%args.batch_size!=0:
        print "Error in load batch_size()"
        quit()
    ################## write txt ##################
    father_path = os.path.abspath(os.path.dirname(args.input_dir) + os.path.sep + "..")
    fp=open(father_path + '/train_pair.txt','w')
    for i in train_txt:
        fp.write(i+'\n')
    fp.close()
    print('finished the train_pair.txt')
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=str, help='Directory with training images.',
                        default = '/home/ajliu/LAJ/ReId_Eigen/datasets/Market-1501-v15.09.15/')
    parser.add_argument('eigen_dir', type=str, help='Directory with eigenbody images.',
                        default = '/home/ajliu/LAJ/ReId_Eigen/datasets/eigen_v2/')
    parser.add_argument('--batch_size', type=int,help='setting batch-size.', default=14)
    parser.add_argument('--is_shuffle', type=bool,help='setting shuffle.', default=False)
    return parser.parse_args(argv)
if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
