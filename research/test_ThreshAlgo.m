% Data
idx_acc = find(type == 'ACC');
raw = value1c(idx_acc);



[h30,t]=impz(fil_30Hz,1);
resampled_30Hz = upfirdn(raw,h30,2,7);

[h3,t] = impz(fil_3hz,1);
resampled_3Hz=upfirdn(resampled_30Hz,h3,1,10);

% Settings
lag = 5;
threshold = 2.0;
influence = 0.15;

% Get results
[signals,avg,dev] = ThresholdingAlgo(resampled_3Hz,lag,threshold,influence);
pk = find(diff(signals)>=1)+1;

close all;
figure; hold on; axis tight; grid on; grid minor;l = legend; set(l, 'interpreter', 'latex');l.Location='best';%subplot(2,1,1); 
x = 1:length(resampled_3Hz); ix = lag+1:length(resampled_3Hz);
area(x(ix),avg(ix)+threshold*dev(ix),'FaceColor',[0.9 0.9 0.9],'EdgeColor','none','HandleVisibility','off');
area(x(ix),avg(ix)-threshold*dev(ix),'FaceColor',[1 1 1],'EdgeColor','none','HandleVisibility','off');

plot(x(ix),avg(ix),'LineWidth',1.5,'Color','k','LineWidth',1.5, 'DisplayName', '$\mu_i$');

plot(x(ix),avg(ix)+threshold*dev(ix),'LineWidth',1,'Color','green','LineWidth',1.5, 'DisplayName', '$\mu_i + t_d\cdot \sigma_i$');
plot(x(ix),avg(ix)-threshold*dev(ix),'LineWidth',1,'Color','green','LineWidth',1.5,'HandleVisibility','off');

plot(x(ix),avg(ix)+0.15,'LineWidth',1,'Color','r','LineWidth',1.5, 'DisplayName', '$t_s$');
plot(x(ix),avg(ix)-0.15,'LineWidth',1,'Color','r','LineWidth',1.5, 'HandleVisibility','off');

plot(1:length(resampled_3Hz),resampled_3Hz,'b','LineWidth',1, 'DisplayName', '3Hz-resampled $acc_z$');
plot(pk, resampled_3Hz(pk),'x','LineWidth',1.5', 'DisplayName', 'Detected Peaks');
xlim([800 1200])
%subplot(2,1,2);
%stairs(signals,'r','LineWidth',1.5); %ylim([-1.5 1.5]);