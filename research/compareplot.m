idx1_acc = find(type == "ACC");
idx1_accf = find(type == "ACCF");
idx1_class = find(type == "CLASS");
idx1_lap = find(type == "LAP");
idx1_stroke = find(type == "STROKE");

idx2_acc = find(type2 == "ACC");
%%
value2d(value2d == 5) = 0
%%
close all;
t = tiledlayout(2,1,'TileSpacing','Compact','Padding','Compact');
nexttile

hold on;axis tight;l = legend; set(l, 'interpreter', 'latex');l.Location='best';grid; grid minor;
plot(ts2(idx2_acc), value2a(idx2_acc), 'LineWidth',1.5, 'DisplayName', 'Original $acc_z (100Hz)$')
plot(ts(idx1_accf), 2.*value1a(idx1_accf), 'LineWidth',1.5, 'DisplayName', 'Filtered $acc_z (30Hz)$')
plot(ts(idx1_lap), -45*ones(1,length(idx1_lap)), 'k^', 'LineWidth',4, 'MarkerSize', 15, 'MarkerFaceColor', 'k', 'DisplayName', 'Recognized laps')
plot(ts(idx1_stroke), 13*ones(1,length(idx1_stroke)), 'rx', 'LineWidth', 4, 'MarkerSize', 10, 'DisplayName', 'Recognized strokes')
xlabel('timestamp [ns]');ylabel('acceleration');ylim([-50,15]);set(gca,'FontSize', 14);
nexttile

hold on;axis tight;l = legend; set(l, 'interpreter', 'latex');l.Location='best';grid; grid minor
plot(ts2, value2d, 'LineWidth',2, 'DisplayName', 'Manual labels')
[M,I] = max([value1a(idx1_class),value1b(idx1_class),value1c(idx1_class),value1d(idx1_class),value1e(idx1_class)],[],2);
plot(ts(idx1_class), I-1, 'LineWidth',2, 'DisplayName', 'SwitPnet output')
xlabel('timestamp [ns]');ylabel('class');
names = {'Null'; 'CR'; 'BR'; 'BK'; 'BY'};
set(gca,'ytick',[0:4],'yticklabel',names);set(gca,'FontSize', 14);

%%
hold on;
plot(ts(idx_accf), 2.*value2a(idx_accf))
plot(ts(idx_class), value2a(idx_class), 'LineWidth',4);
plot(ts(idx_lap), zeros(1,length(ts2(idx_lap))), 'x', 'LineWidth',4)
