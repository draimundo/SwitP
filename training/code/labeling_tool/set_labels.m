%% 1. Import data

if exist('step4','var')
    error('Are you sure you want to restart? Didnt export data!');
end

clear all
close all

recording_date = '2020_03_05';
user = 'felix';
num = 1;

save_folder_path = fullfile('..\..\data\re_labeled');
path_to_recordings = fullfile('..\..\data\raw', recording_date, 'sorted', user);

csv_in_path = dir(fullfile(path_to_recordings, '*.csv'));
Mfull = importdata(fullfile(path_to_recordings, csv_in_path(num).name));

starttime_str = regexp(csv_in_path.name, '\d*','Match');
starttime = str2double(starttime_str)*1E-9; %Note how MATLAB defines posix time in seconds...

head = Mfull.textdata{1};
flag = Mfull.textdata{end};

values = Mfull.data;
timestamps_cells = Mfull.textdata(2:(end-1),1);
sensors = Mfull.textdata(2:(end-1), 2);
timestamps_double = str2double(timestamps_cells)...
                    -str2double(timestamps_cells(1))...
                    +starttime*1E9; % Convert system uptime in ns to POSIX time (hoping first meas when app started?!


timestamps = datetime(timestamps_double*1E-9, 'ConvertFrom', 'posixtime'); %Note how MATLAB defines posix time in seconds...

acc_rows = cellfun(@(x) strcmp(x, 'ACC'), sensors);
acc_values = values(acc_rows,:);
acc_timestamps_double = timestamps_double(acc_rows);
acc_timestamps = datetime(timestamps(acc_rows), 'Format', 'HH:mm:ss');

%% 2. Select bounds (can be re-run or datatips deleted in figure) DON'T CLOSE if you want to keep the markers!!!
% Use datatips to select bounds between swimtypes
% Hold ALT to have multiple datatips
% Can move with arrows
h = figure; hold all;
title(sprintf('User: %s, Date: %s, Label: %s, Flag: %s', user, recording_date, head, flag))
plot(acc_timestamps, acc_values(:,1), 'linewidth', 1)
plot(acc_timestamps, acc_values(:,2), 'linewidth', 1)
plot(acc_timestamps, acc_values(:,3), 'linewidth', 1)
xlim([acc_timestamps(1) acc_timestamps(1)+minutes(1)]) %default zoom on a minute =~50m
box on;
disp('Pick points')

dcm_obj = datacursormode(h);
set(dcm_obj,'DisplayStyle','datatip','SnapToDataVertex','off','Enable','on');

%% 3. Get cursor info 
c_info = getCursorInfo(dcm_obj);
close all;

if isempty(c_info)
    error('no cursors in plot - redo step 2!')
end
%% 4. Plot selected intervals, and select classes (can be re-run if mistake in class-choosing)
% If fenster closed - aborted
close all;

cursor_dataindices = arrayfun(@(x) x.DataIndex, c_info);
cursor_dataindices = [1, sort(cursor_dataindices), length(acc_timestamps)];
window_starts = cursor_dataindices(1:(end-1));
true_window_starts = zeros(size(window_starts));
window_stops = cursor_dataindices(2:end);
true_window_stops = zeros(size(window_stops));

window_label = zeros(length(window_starts),1);
data_label = nan(length(acc_timestamps),1);

acc_timestamps_label = ones(length(acc_timestamps_double), 1).*-1; %Assign everything to unknown by default
timestamps_label = ones(length(timestamps_double), 1).*-1; %Assign everything to unknown by default

stylelist = {'unknown', 'null', 'freestyle', 'breaststroke', 'backstroke', 'butterfly', 'turn', 'kick', 'RETRY LAST STEP'};

i = 1;
while i <= length(window_starts) %MATLAB doesn't support changing index in for
    h = figure;
    sp(1) = subplot(3,1,1); hold all;
	this_window_ix = window_starts(i):window_stops(i);
    plot(acc_timestamps, acc_values(:,1), 'b', 'linewidth', 1)
    plot(acc_timestamps, acc_values(:,2), 'r', 'linewidth', 1)
    plot(acc_timestamps, acc_values(:,3), 'g', 'linewidth', 1)
    
    plot(acc_timestamps(this_window_ix), acc_values(this_window_ix,1), 'b', 'linewidth', 3)
    plot(acc_timestamps(this_window_ix), acc_values(this_window_ix,2), 'r', 'linewidth', 3)
    plot(acc_timestamps(this_window_ix), acc_values(this_window_ix,3), 'y', 'linewidth', 3)
    axis tight
    
    sp(2) = subplot(3,1,2); hold all;
    plot(acc_timestamps(this_window_ix), acc_values(this_window_ix,1), 'b', 'linewidth', 3)
    plot(acc_timestamps(this_window_ix), acc_values(this_window_ix,2), 'r', 'linewidth', 3)
    plot(acc_timestamps(this_window_ix), acc_values(this_window_ix,3), 'g', 'linewidth', 3)
    if i>1
        prev_window_ix = window_starts(i-1):window_stops(i-1);
        plot(acc_timestamps(prev_window_ix), acc_values(prev_window_ix,1), 'b')
        plot(acc_timestamps(prev_window_ix), acc_values(prev_window_ix,2), 'r')
        plot(acc_timestamps(prev_window_ix), acc_values(prev_window_ix,3), 'g')
    end
    if i<length(window_starts)
        next_window_ix = window_starts(i+1):window_stops(i+1);
        plot(acc_timestamps(next_window_ix), acc_values(next_window_ix,1), 'b')
        plot(acc_timestamps(next_window_ix), acc_values(next_window_ix,2), 'r')
        plot(acc_timestamps(next_window_ix), acc_values(next_window_ix,3), 'g')
    end
    title(sprintf('Window %d', i))
    axis tight
    
    sp(3) = subplot(3,1,3); hold all;
    plot(acc_timestamps(this_window_ix), acc_values(this_window_ix,1), 'b', 'linewidth', 3)
    plot(acc_timestamps(this_window_ix), acc_values(this_window_ix,2), 'r', 'linewidth', 3)
    plot(acc_timestamps(this_window_ix), acc_values(this_window_ix,3), 'g', 'linewidth', 3)
    title(sprintf('Window %d', i))
    axis tight
    
    h.WindowState = 'maximized'; %Maximize window
    
    [indx,tf] = listdlg('PromptString', {'Select style'}, 'SelectionMode','single','ListString',stylelist);
    if tf
        if indx == length(stylelist) % User wants to redo last step, not a class
            i = i-1;
            close(h)
            continue
        else
            window_label(i) = indx-2;
        end
    else
        error('Nothing selected - aborting');
    end

    true_window_starts(i) = find(timestamps_double >= acc_timestamps_double(this_window_ix(1)),1,'first'); %first element in window
    true_window_stops(i) = find(timestamps_double < acc_timestamps_double(this_window_ix(end)),1,'last'); %last element not considered in window
    
    acc_timestamps_label(this_window_ix) = window_label(i);
    timestamps_label(true_window_starts(i):true_window_stops(i)) = window_label(i);
    
    close(h)
    i = i+1;
end

step4 = true;

%% 5. Plot to verify selection

label_colour = ['y', 'b', 'r', 'g', 'm', 'c', 'k', 'c']; %kick same color as butterfly!
unique_labels = unique(acc_timestamps_label);
h = figure; 
sp(1) = subplot(3,1,1); hold all
for i=1:length(unique_labels)
    ix = find(acc_timestamps_label==unique_labels(i));
    plot(acc_timestamps(ix), acc_values(ix, 1), 'Color', label_colour(unique_labels(i)+2))
end
sp(2) = subplot(3,1,2); hold all
for i=1:length(unique_labels)
    ix = find(acc_timestamps_label==unique_labels(i));
    plot(acc_timestamps(ix), acc_values(ix, 2), 'Color',label_colour(unique_labels(i)+2))
end
sp(3) = subplot(3,1,3); hold all
for i=1:length(unique_labels)
    ix = find(acc_timestamps_label==unique_labels(i));
    plot(acc_timestamps(ix), acc_values(ix, 3), 'Color',label_colour(unique_labels(i)+2))
end
linkaxes(sp, 'x')

%% 6. Write to file (keeping the old folderstyle for compatibility)

mkdir(fullfile(save_folder_path, recording_date, user))
save_file_path = fullfile(save_folder_path, recording_date, user, csv_in_path(num).name);

f = fopen(save_file_path, 'w');
fprintf(f, sprintf('%s\n', head));
for i=1:length(timestamps_double)
    line = sprintf('%s; %s; %f; %f; %f; %d\n', timestamps_cells{i}, sensors{i}, ...
           values(i,1), values(i,2), values(i,3), timestamps_label(i));
    fprintf(f, line);
end
eoftag = sprintf('%s\n', flag);
fprintf(f, eoftag);
fclose(f);

clear step4