%%%%%%%%%%%%%% Etract IDE features %%%%%%%%%%%%%%%%
% clc;clear all;
% netname = 'res50_s';
% root_path = '/home/ajliu/LAJ/ReId_Eigen'
% fea_mat_path = [root_path '/out_features'];
% is_extracting = true;
% type_fea = 'python' ;%'python' 'matlab'
% 
% gpu = 1;
% workspace_root = [root_path];
% caffe_root = [root_path '/caffe-DDM'];
% max_it = '15000';
% model = [workspace_root '/' netname '/deploy.prototxt'];
% weights = [workspace_root '/' netname '/snapshot/' netname '_iter_' max_it '.caffemodel'];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fp=fopen([root_path '/results.txt'],'a');
data_root = [root_path '/datasets/Market-1501-v15.09.15'];
type_fea_path = fea_mat_path;
%%%%%%%%%%%%%%
addpath(genpath('utils/'));
ef_name = {'test.lst', 'query.lst'};
if is_extracting
    addpath([caffe_root, '/matlab/']);
    % extract features
    caffe.set_device(gpu);
    caffe.set_mode_gpu();
    net = caffe.Net(model, weights, 'test');
    if strcmp(netname(1:5), 'res50')
        im_size = 224;
        feat_dim = 2048;
        im_height = 224;
        im_width = 224;
   elseif strcmp(netname(1:5), 'res18')
        im_size = 112;
        feat_dim = 2048;
        im_height = 112;
        im_width = 112;
    elseif strcmp(netname, 'CaffeNet')
        im_size = 227;
        feat_dim = 4096;
    else
        im_size = 227;
        feat_dim = 1024;
    end
    mean_data = importdata([caffe_root,'/matlab/+caffe/imagenet/ilsvrc_2012_mean.mat']);
    image_mean = mean_data;
    off = floor((size(image_mean,1) - im_size)/2)+1;
    image_mean = image_mean(off:off+im_size-1, off:off+im_size-1, :);
    ef_path = {[data_root, '/bounding_box_test/'], [data_root,'/query/']};
    if ~exist(type_fea_path)
        mkdir(type_fea_path)
    end
    for i = 1:2
        img_path = ef_path{i};
        img_file = dir([img_path '*.jpg']);
        feat = single(zeros(feat_dim, length(img_file)));
        for n = 1:length(img_file)
            img_name = [img_path  img_file(n).name];
            im = imread(img_name);
            im_1 = prepare_img( im, image_mean, im_size);
            feat_img = net.forward({im_1});
            feat(:, n) = single(feat_img{1}(:));
        end
        if strcmp(type_fea,'matlab')==1
            save([type_fea_path '/' netname  '_' type_fea '_' ef_name{i} 'feature.mat'],'feat');
        else
            fprint('your type_fea is wrrong!');
            exit
        end
        if i==1 %test
            galFea = feat;
        else %query
            probFea = feat;
        end
        feat = [];
    end
    caffe.reset_all();
else
    if strcmp(type_fea,'python')
        galFea =  importdata([type_fea_path '/' netname '_' type_fea '_' ef_name{1} '.feature.mat'])';
        probFea = importdata([type_fea_path '/' netname '_' type_fea '_' ef_name{2} '.feature.mat'])';
    elseif strcmp(type_fea,'matlab')
        galFea =  importdata([type_fea_path '/' netname '_'  type_fea '_' ef_name{1} 'feature.mat']);
        probFea= importdata([type_fea_path '/' netname '_'  type_fea '_' ef_name{2} 'feature.mat']);
    else
        fprint('your type_fea is wrrong!');
        exit
    end
end
%%%%%%%%%%%% Testing %%%%%%%%%%
%% normalize
sum_val = sqrt(sum(galFea.^2));
for n = 1:size(galFea, 1)
    galFea(n, :) = galFea(n, :)./sum_val;
end

sum_val = sqrt(sum(probFea.^2));
for n = 1:size(probFea, 1)  
    probFea(n, :) = probFea(n, :)./sum_val;
end

%% test info
galFea = single(galFea);
probFea = single(probFea);
test_mat_path  = fullfile(data_root, '/testData.mat');
query_mat_path = fullfile(data_root, '/queryData.mat');
test_mat       = load(test_mat_path);
query_mat      = load(query_mat_path);
nQuery = length(query_mat.query_files);
nTest  = length(test_mat.test_files);
assert(nQuery == size(probFea, 2));
assert(nTest == size(galFea, 2));
assert (all(query_mat.queryCAM >= 1)); assert (all(query_mat.queryCAM <= 6));
assert (all(test_mat.testCAM >= 1));   assert (all(test_mat.testCAM <= 6));
label_gallery   = test_mat.testID;
label_query  = query_mat.queryID;
cam_gallery  = test_mat.testCAM;
cam_query = query_mat.queryCAM;
fprintf('Load data and features done.\n');
%%%%%%%%%%%%%%%% Multi-query
my_pdist2 = @(A, B) sqrt( bsxfun(@plus, sum(A.^2, 2), sum(B.^2, 2)') - 2*(A*B'));
dist_eu = my_pdist2(galFea', probFea');
[CMC_eu, map_eu, ~, ~] = evaluation(dist_eu, label_gallery, label_query, cam_gallery, cam_query);
fprintf(['Multi-query: The (' netname ') + Euclidean performance:\n']);
fprintf('Rank1,   mAP\n');
fprintf('%5.2f%%, %5.2f%%\n\n', CMC_eu(1) * 100, map_eu(1)*100);

fprintf(fp, '\n\n%s: %s  %s + %s  %s:\n','Multi-query','The',netname,type_fea,'Euclidean performance');
fprintf(fp,'%s\n','Rank1,   mAP');
fprintf(fp,'%5.2f%%, %5.2f%%\n', CMC_eu(1) * 100, map_eu(1)*100);
fclose(fp);
exit
