function out = RSNR(ref,tar)
%--------------------------------------------------------------------------
% The reconstruction SNR (RSNR)
%
%--------------------------------------------------------------------------

out = 10*log(sum(sum(sum((ref).^2)))/sum(sum(sum((tar-ref).^2))));