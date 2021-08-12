clear all;
close all;
clc;

%Set the optimal lambda value here
SS_Lambda = 0.8;

%Other parameters...
N_total     = 81;
SCALE       = 4;

% Order = [Nearest, Bicubic, Lapsrn, dip_s, dip_ss]
CC_val      = zeros(1,5);
ERGAS_val   = zeros(1,5);
RMSE_val    = zeros(1,5);
RSNR_val    = zeros(1,5);
SAM_val     = zeros(1,5);
PSNR_val     = zeros(1,5);
for N = 1:N_total
    % Comparision of with and without spatial energy term
    fp_dhp_ss           = strcat("./chikusei/chikusei_",num2str(N, "%02d"),"/chikusei_",num2str(N, "%02d"),"_dhp_",num2str(SS_Lambda*10),".mat");
    fp_dhp_spectral     = strcat("./chikusei/chikusei_",num2str(N, "%02d"),"/chikusei_",num2str(N, "%02d"),"_dhp_0.mat");
    fp_lapsrn           = strcat("./chikusei/chikusei_",num2str(N, "%02d"),"/chikusei_",num2str(N, "%02d"),"_lapsrn.mat");
    fp_input            = strcat("./chikusei/chikusei_",num2str(N, "%02d"),"/chikusei_",num2str(N, "%02d"),".mat");
    
   % Loading all the results
    load(fp_dhp_ss);
    dhp_ss = double(dhp);

    load(fp_dhp_spectral);
    dhp_s = double(dhp);

    load(fp_lapsrn);
    lapsrn = double(lapsrn);
    load(fp_input);

    bicubic = imresize(y, SCALE, "bicubic");
    nearest = imresize(y, SCALE, "nearest");
    
    %Computing performance metrics
    ref     = ref./max(ref, [], "all");
    nearest = nearest./max(nearest, [], "all");
    bicubic = bicubic./max(bicubic, [], "all");
    lapsrn  = lapsrn./max(lapsrn, [], "all");
    dhp_s   = dhp_s./max(dhp_s, [], "all");
    dhp_ss  = dhp_ss./max(dhp_ss, [], "all");
    
    CC_val      = CC_val + [mean(CC(ref, nearest)), mean(CC(ref, bicubic)), mean(CC(ref, lapsrn)), mean(CC(ref, dhp_s)), mean(CC(ref, dhp_ss))];
    ERGAS_val   = ERGAS_val + [ERGAS(ref, nearest, SCALE), ERGAS(ref, bicubic, SCALE), ERGAS(ref, lapsrn, SCALE), ERGAS(ref, dhp_s, SCALE), ERGAS(ref, dhp_ss, SCALE)];
    RMSE_val    = RMSE_val + [RMSE(ref, nearest), RMSE(ref, bicubic), RMSE(ref, lapsrn), RMSE(ref, dhp_s), RMSE(ref, dhp_ss)];
    RSNR_val    = RSNR_val + [RSNR(ref, nearest), RSNR(ref, bicubic), RSNR(ref, lapsrn), RSNR(ref, dhp_s), RSNR(ref, dhp_ss)];
    SAM_val     = SAM_val + [SAM(ref, nearest), SAM(ref, bicubic), SAM(ref, lapsrn), SAM(ref, dhp_s), SAM(ref, dhp_ss)];
    PSNR_val    = PSNR_val + [PSNR(nearest, ref), PSNR(bicubic, ref), PSNR(lapsrn, ref), PSNR(dhp_s, ref), PSNR(dhp_ss, ref)];
end

CC_val/N_total
SAM_val/N_total
RMSE_val/N_total
RSNR_val/N_total
ERGAS_val/N_total
PSNR_val/N_total