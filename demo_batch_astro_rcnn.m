clear all;
% Add the ImageGraph to path (in a folder named 'ImageGraphs'). Find it here:
% http://www.mathworks.com/matlabcentral/fileexchange/53614-image-graphs
addpath(fullfile(fileparts(mfilename('fullpath')), 'ImageGraphs'));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% read lesion ground truth and combine
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% them
mattingfolder = 'C:\data\image_matting\SemanticSoftSegmentation-master\maskrcnn\new\valattention-test\';
testingfolder = 'C:\data\image_matting\SemanticSoftSegmentation-master\maskrcnn\new\testoutput\';
datadirs_pt = dir(mattingfolder);
dircell_pt = struct2cell(datadirs_pt)' ;
filenames_pt = dircell_pt(:,1);

for pt_i =3:size(dircell_pt,1) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% for pt_i = 7:7  
ptfolder = [mattingfolder filenames_pt{pt_i,1} '\'];
subfolder = [ptfolder '\*_new.png']; %%%%
datadirs = dir(subfolder);
dircell = struct2cell(datadirs)';
filenames = dircell(:,1);

%% Read the image and features from the sample file
% lsfolder = 'C:\data\image_matting\SemanticSoftSegmentation-master\improd\mal-TA\layer17\';
% for i=1:size(dircell,1) %% pt index
for i=4:4
     %%%%%%%%%%%%%%%%%%%%%%%      
   pred3d = single(zeros(256, 256, size(dircell,1)));
   idx = regexp(filenames{i,1}, '_', 'split'); %% string splitting
   image_path = [ptfolder filenames{i,1}];   
   image = im2double(imread(image_path));  %%% replace to image-of-interest
%%%%%%%%%%%%%%%%%%%%%%%%
features = image(:, size(image, 2) / 2 + 1 : end, :);
image = image(:, 1 : size(image, 2) / 2, :);

% The eigendecomposition uses a lot of memory and may render the computer
% unresponsive, so better to test it first with a small image.
image = imresize(image, 0.5);
features = imresize(features, 0.5);

%% Semantic soft segmentation
% This function outputs many intermediate variables, if needed.
% The results may vary a bit from run to run, as there are 2 stages that use 
% k-means for intialization & grouping.
sss = SemanticSoftSegmentation(image, features);

% To use the features generated using our network implementation,
% just feed them as the 'features' variable to the function. It will do
% the prepocessing described in the paper and give the processed
% features as an output.
% If you are dealing with many images, storing the features after
% preprocessing is recommended as raw hyperdimensional features
% take a lot of space. Check the 'preprocessFeatures.m' file.

% Visualize
% figure; 
imshow([image features visualizeSoftSegments(sss)]);
title('Semantic soft segments');
% % ss_path = [folderdir idx{1,1} '_ss.png'];
ss_path = [ptfolder idx{1,2} '_ss.png'];
saveas(gcf, ss_path);

% % There's also an implementation of Spectral Matting included
% sm = SpectralMatting(image);
% You can group res, the way we presented our comparisons in the paper.
% sm_gr = groupSegments(sm, features);
% figure; imshow([image visualizeSoftSegments(sm) visualizeSoftSegments(sm_gr)]);
% title('Matting components');
% figure
% imshow([sss(:,:,1) sss(:,:,2) sss(:,:,3) sss(:,:,4) sss(:,:,5) sss(:,:,6) sss(:,:,7) sss(:,:,8)])
% sss_path = [ptfolder idx{1,1} '_sss.png'];
% saveas(gcf, sss_path);

%%%%%%
dice = zeros(size(sss,3),1);
sens = zeros(size(sss,3),1);
%%%%%%%%%%%
%%%%%%%%%%%%%
real_path = [ptfolder '1_' idx{1,2} '_roi_real.png'];
real = imread(real_path);
real = single(real);
real = real/max(real(:));
%%%%%%%%%%%%%%%%%%%%


for k=1:size(sss,3)    
% pred = imresize(sss(:,:,k),[size(pm_crop(:,:,str2num(idx{1,1})),1) size(pm_crop(:,:,str2num(idx{1,1})),2)]).* bwmorph(pm_crop(:,:,str2num(idx{1,1})),'thin',1); %%%% prediction
pred = imresize(sss(:,:,k),[256 256]);
% pred = (pred> prctile(pred(:),75));
pred = single(pred);

% if any(gt(:))
% common = (pred & gt);
% b = sum(gt(:));
% else
    common = (pred & real);
    b = sum(real(:));
% end
a = sum(common(:));
c = sum(pred(:));
dice(k) = 2*a/(b+c);
sens(k) = a/b;
end
[aa,bb] = max(dice);
%%%%%%%%
% pred = imresize(sss(:,:,bb),[size(pm_crop(:,:,str2num(idx{1,1})),1) size(pm_crop(:,:,str2num(idx{1,1})),2)]).*  bwmorph(pm_crop(:,:,str2num(idx{1,1})),'thin',1); %%%% prediction
if aa < 0.1 %%%% false positive slice
    pred = single(real);
else
pred = imresize(sss(:,:,bb),[256 256]);
% pred = (pred> prctile(pred(:),75));
pred3d(:,:,i) = single(pred);
end
end



for i=1:size(dircell,1) %% pt index

     %%%%%%%%%%%%%%%%%%%%%%%      
   pred = (pred3d(:,:,i)> prctile(pred3d(:),25));
   idx = regexp(filenames{i,1}, '_', 'split'); %% string splitting
% %%%%%%%%%% filling holes
% CC = bwconncomp(pred);
% conn_count = zeros(CC.NumObjects, 1);
% for conn_idx = 1:CC.NumObjects
%     pred_temp = zeros(size(pred,1), size(pred,2));
%     pred_temp(CC.PixelIdxList{conn_idx}) = 1;
%     pred_temp=imfill(pred_temp, 'holes');
%     clear common_connc;
%     common_connc = (pred_temp & real);
%     conn_count(conn_idx) = sum(common_connc(:));
% end
% [aaa bbb] = max(conn_count);
% %%%%%%%%%%%%%%%
% pred = zeros(size(pred,1), size(pred,2));
% pred(CC.PixelIdxList{bbb}) = 1;
% pred=imfill(pred, 'holes');
% %%%%%%%%%%%%%%%%%%%%%%%5
%%%%%%%%%%%
gt_path = [ptfolder '1_' idx{1,2} '_gt.png'];
if exist(gt_path, 'file') == 2
gt = imread(gt_path);
gt = imresize(gt,[256 256]);
gt = gt/max(gt(:));
else
    gt = zeros(256, 256);
end
% S = regionprops(gt,'Centroid');
%%%%%%%%%%%%%
real_path = [ptfolder '1_' idx{1,2} '_roi_real.png'];
real = imread(real_path);
real = single(real);
real = real/max(real(:));
%%%%%%%%%%%%%%%%%%%%
pm_path = [ptfolder '1_' idx{1,2} '_pm.png'];
pm = imread(pm_path);
pm1 = double(pm(:,:,1)); % 3 channels to 1 channel and unit to double
% pm = imresize(pm,[256 256]);
pm1 = pm1/max(pm1(:));
pm1 = bwmorph(pm1,'thin',1);
pm2 = bwmorph(real,'thick',7);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
pred = pred.* pm1 .* pm2;
%%%%%%%%%%%%% recalculate dice
common = (pred & gt);
a = sum(common(:));
b = sum(gt(:));
c = sum(pred(:));
dice_last = 2*a/(b+c);
sens_last = a/b;
% %%%%%%%%%%%
imshow([gt pred],[]);
dice_display = sprintf('Dice: %.2f, \n Sensitivity: %.2f', dice_last, sens_last);
xlabel(dice_display, 'FontSize', 25);
dice_path = [ptfolder idx{1,2} '_dice.png'];
saveas(gcf, dice_path);
final_path = [ptfolder idx{1,2} '_matting.png'];
imwrite(pred, final_path);
% test_path = [testingfolder filenames_pt{pt_i,1} '-' idx{1,2} '.png'];
% imwrite(pred, test_path);
end
     
end

