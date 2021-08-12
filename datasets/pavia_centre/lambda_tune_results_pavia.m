clear all
close all
clc

% Select optimal value of $\lambda$ for DIP process
lambda = [0.0, 0.1, 0.4, 0.6, 0.8, 0.9, 1.0];
SCALE = 4;
N_patches = 24;

cc_v    = zeros(1, length(lambda));
sam_v   = zeros(1, length(lambda));
rmse_v  = zeros(1, length(lambda));
ergas_v = zeros(1, length(lambda));
psnr_v  = zeros(1, length(lambda));
    
for i =  1: 1:length(lambda)
    for n=1:N_patches
        % File paths...
        fp_input         = strcat("./pavia/pavia_",num2str(n, "%02d"),"/pavia_",num2str(n, "%02d"),".mat");
        file_name_lambda = strcat("./pavia/pavia_",num2str(n, "%02d"),"/pavia_",num2str(n, "%02d"),"_dhp_",num2str(lambda(i)*10),".mat");
        
        %Load files:
        load(fp_input);
        load(file_name_lambda);
        dhp = double(dhp);
        
        %Computing performance metrics
        ref = ref./max(ref, [], "all");
        dhp = dhp./max(dhp, [], "all");
        cc_v(i)      = cc_v(i)      + mean(CC(ref, dhp));
        ergas_v(i)   = ergas_v(i)   + ERGAS(ref, dhp, SCALE);
        rmse_v(i)    = rmse_v(i)    + RMSE(ref, dhp);
        sam_v(i)     = sam_v(i)     + SAM(ref, dhp);
        psnr_v(i)    = psnr_v(i)    + PSNR(dhp, ref);
    end
end
cc_v    = cc_v./N_patches;
ergas_v = ergas_v./N_patches;
rmse_v  = rmse_v./N_patches;
sam_v   = sam_v./N_patches;
psnr_v  = psnr_v./N_patches;


%% -------------         Generating the figure  ----------------------------%
close all
fig = figure("Position", [100, 100, 1500, 200]);
axes
axis('equal')  % [EDITED: or better 'square' !]

subplot(1,5,1);
plot(lambda, cc_v, 'k*-','LineWidth',2);
title("CC (\uparrow)")
grid on
xlabel("\lambda", 'FontSize', 24)
ylabel("CC value", 'FontSize', 20)
xticks(lambda)
xlim([0, lambda(end)+0.01])

subplot(1,5,2);
hold on
title("SAM (\downarrow)");
grid on;
plot(lambda, sam_v, 'k*-','LineWidth',2);
grid on
xlabel("\lambda", 'FontSize', 24)
ylabel("SAM value", 'FontSize', 20)
xticks(lambda)
xlim([0, lambda(end)+0.01])

subplot(1,5,3);
plot(lambda, rmse_v, 'k*-','LineWidth',2);
title("RMSE (\downarrow)")
grid on
xlabel("\lambda", 'FontSize', 24)
ylabel("RMSE value", 'FontSize', 20)
xticks(lambda)
xlim([0, lambda(end)+0.01])

subplot(1,5,4);
plot(lambda, ergas_v, 'k*-','LineWidth',2);
title("ERGAS (\downarrow)")
grid on
xlabel("\lambda", 'FontSize', 24)
ylabel("ERGAS value", 'FontSize', 20)
xticks(lambda)
xlim([0, lambda(end)+0.01])

subplot(1,5,5);
plot(lambda, psnr_v, 'k*-','LineWidth',2);
title("PSNR (\uparrow)")
grid on
xlabel("\lambda", 'FontSize', 24)
ylabel("PSNR value", 'FontSize', 20)
xticks(lambda)
xlim([0, lambda(end)+0.01])

set(gcf,'Units','inches');
screenposition = get(gcf,'Position');
set(gcf,...
    'PaperPosition',[-1.74 0.00 18.68 2.50],...
    'PaperSize',[15.50, 2.50]);
%---------------------- Saving to DropBox --------------------------------%
saveas(gcf,'/home/lidan/Dropbox/Apps/Overleaf/IEEEGR_V1/imgs/pavia_lambda.pdf')
% clc;
% close all;