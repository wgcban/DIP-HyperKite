% Author: Chaminda Bandara (wbandar1@jhu.edu/
% chaminda.bandara@eng.pdn.ac.lk)
% Date: 17-June-2021

% Plot comparisions with different upsampling methods
clear all;
close all;
clc;

%Set optimal value here
SS_Lambda = 0.8;

%% FIRST PATCH
% Comparision of with and without spatial energy term
fp_dhp_ss           = strcat("./botswana/botswana_12/botswana_12_dhp_",num2str(SS_Lambda*10),".mat");
fp_dhp_spectral     = "./botswana/botswana_12/botswana_12_dhp_0.mat";
fp_lapsrn           = "./botswana/botswana_12/botswana_12_lapsrn.mat";
fp_input            = "./botswana/botswana_12/botswana_12.mat";

% Loading all the results
load(fp_dhp_ss);
dhp_ss = dhp;

load(fp_dhp_spectral);
dhp_s = dhp;

load(fp_lapsrn);
load(fp_input);

%Different up-sampling techniques
rgb_y       = get_rgb(y);
rgb_bicubic = get_rgb(imresize(y, 3, "bicubic"));
rgb_nearest = get_rgb(imresize(y, 3, "nearest"));
rgb_lapsrn 	= get_rgb(lapsrn);
rgb_dhp_ss 	= get_rgb(dhp_ss);
rgb_dhp_s 	= get_rgb(dhp_s);
rgb_ref     = get_rgb(ref);

%Amplification factor 
a = 1.5;

fig = figure("Position", [100, 100, 1500, 200]);
axes

subplot(1,7,1);
hold on;
title("(a)",'FontSize',12)
imshow(a*rgb_y)
axis('equal')

subplot(1,7,2);
hold on;
title("(b)",'FontSize',12)
imshow(a*rgb_nearest)
axis('equal')

subplot(1,7,3);
hold on;
title("(c)",'FontSize',12)
imshow(a*rgb_bicubic)
axis('equal')

subplot(1,7,4);
hold on
title("(d)",'FontSize',12)
imshow(a*rgb_lapsrn)
axis('equal')

subplot(1,7,5);
hold on
title("(e)",'FontSize',12)
imshow(a*rgb_dhp_s)
axis('equal')

subplot(1,7,6);
hold on
title("(f)",'FontSize',12)
imshow(a*rgb_dhp_ss)
axis('equal')

subplot(1,7,7);
hold on
title("(g)",'FontSize',12)
imshow(a*rgb_ref)
axis('equal')


set(gcf,'Units','inches');
screenposition = get(gcf,'Position');
set(gcf,...
    'PaperPosition',[-1.74 0.00 18.68 2.50],...
    'PaperSize',[15.50, 2.50]);
%---------------------- Saving to DropBox --------------------------------%
saveas(gcf,'/home/lidan/Dropbox/Apps/Overleaf/IEEEGR_V1/imgs/botswana_upsampling1.pdf')

imwrite(a*rgb_y, './botswana_up_rgb_comp/1_y.png')
imwrite(a*rgb_nearest, './botswana_up_rgb_comp/1_nearest.png')
imwrite(a*rgb_bicubic, './botswana_up_rgb_comp/1_bicubic.png')
imwrite(a*rgb_lapsrn, './botswana_up_rgb_comp/1_lapsrn.png')
imwrite(a*rgb_dhp_s, './botswana_up_rgb_comp/1_dhp_s.png')
imwrite(a*rgb_dhp_ss, './botswana_up_rgb_comp/1_dhp_ss.png')
imwrite(a*rgb_ref, './botswana_up_rgb_comp/1_rgb_ref.png')


%% SECOND PATCH
% Comparision of with and without spatial energy term
fp_dhp_ss           = strcat("./botswana/botswana_14/botswana_14_dhp_",num2str(SS_Lambda*10),".mat");
fp_dhp_spectral     = "./botswana/botswana_14/botswana_14_dhp_0.mat";
fp_lapsrn           = "./botswana/botswana_14/botswana_14_lapsrn.mat";
fp_input            = "./botswana/botswana_14/botswana_14.mat";

% Loading all the results
load(fp_dhp_ss);
dhp_ss = dhp;

load(fp_dhp_spectral);
dhp_s = dhp;

load(fp_lapsrn);
load(fp_input);

%Different up-sampling techniques
rgb_y       = get_rgb(y);
rgb_bicubic = get_rgb(imresize(y, 3, "bicubic"));
rgb_nearest = get_rgb(imresize(y, 3, "nearest"));
rgb_lapsrn 	= get_rgb(lapsrn);
rgb_dhp_ss 	= get_rgb(dhp_ss);
rgb_dhp_s 	= get_rgb(dhp_s);
rgb_ref     = get_rgb(ref);

%Amplification factor 
a = 1.5;

% Plot all the results
fig = figure("Position", [100, 100, 1500, 200]);
axes

subplot(1,7,1);
hold on;
% title("(a)",'FontSize',12)
imshow(a*rgb_y)
axis('equal')

subplot(1,7,2);
hold on;
% title("(b)",'FontSize',12)
imshow(a*rgb_nearest)
axis('equal')

subplot(1,7,3);
hold on;
% title("(c)",'FontSize',12)
imshow(a*rgb_bicubic)
axis('equal')

subplot(1,7,4);
hold on
% title("(d)",'FontSize',12)
imshow(a*rgb_lapsrn)
axis('equal')

subplot(1,7,5);
hold on
% title("(e)",'FontSize',12)
imshow(a*rgb_dhp_s)
axis('equal')

subplot(1,7,6);
hold on
% title("(f)",'FontSize',12)
imshow(a*rgb_dhp_ss)
axis('equal')

subplot(1,7,7);
hold on
% title("(g)",'FontSize',12)
imshow(a*rgb_ref)
axis('equal')


set(gcf,'Units','inches');
screenposition = get(gcf,'Position');
set(gcf,...
    'PaperPosition',[-1.74 0.00 18.68 2.50],...
    'PaperSize',[15.50, 2.50]);
%---------------------- Saving to DropBox --------------------------------%
saveas(gcf,'/home/lidan/Dropbox/Apps/Overleaf/IEEEGR_V1/imgs/botswana_upsampling2.pdf')

imwrite(a*rgb_y, './botswana_up_rgb_comp/2_y.png')
imwrite(a*rgb_nearest, './botswana_up_rgb_comp/2_nearest.png')
imwrite(a*rgb_bicubic, './botswana_up_rgb_comp/2_bicubic.png')
imwrite(a*rgb_lapsrn, './botswana_up_rgb_comp/2_lapsrn.png')
imwrite(a*rgb_dhp_s, './botswana_up_rgb_comp/2_dhp_s.png')
imwrite(a*rgb_dhp_ss, './botswana_up_rgb_comp/2_dhp_ss.png')
imwrite(a*rgb_ref, './botswana_up_rgb_comp/2_rgb_ref.png')