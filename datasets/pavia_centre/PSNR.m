function [mean_psnr] = PSNR(pred,ref)
% pred: predicted
% ref: reference

l = size(ref, 3);
mean_psnr = 0.0;
for i =1:1:l
    mean_psnr = mean_psnr + (1/l)*psnr(pred(:,:,i), ref(:,:,i));
end
end

