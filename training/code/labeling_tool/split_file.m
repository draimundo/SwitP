%% 1. Import data
if exist('step4','var')
    error('Are you sure you want to restart? Didnt export data!');
end

clear all
close all

recording_date = '2020_03_05';
watch_number = 'N1';
num = 1;
user = 'unknownman';
bSort = true; % sort by timestamp

save_folder_root = fullfile('..\..\data\raw', recording_date);
path_to_recordings = fullfile(save_folder_root, watch_number);
csv_in_path = dir(fullfile(path_to_recordings, '*.csv'));
Mfull = importdata(fullfile(path_to_recordings, csv_in_path(num).name));

starttime_str = regexp(csv_in_path.name, '\d*','Match');
starttime = str2double(starttime_str)*1E-3; %Note how MATLAB defines posix time in seconds...

head = Mfull.textdata{1};
flag = Mfull.textdata{end,1};

values = Mfull.data;
timestamps_cells = Mfull.textdata(2:(end-1),1);
sensors = Mfull.textdata(2:(end-1), 2);
timestamps_double = str2double(timestamps_cells)...
                    -str2double(timestamps_cells(1))...
                    +starttime*1E9; % Convert system uptime in ns to POSIX time (hoping first meas when app started?!

% Sometimes timestamps are not strictly increasing, better for cutting all
% sensors in the good places
if bSort
    [timestamps_double, I] = sort(timestamps_double);
    timestamps_cells = timestamps_cells(I);
    values = values(I,:);
    sensors = sensors(I,:);
end
    
timestamps = datetime(timestamps_double*1E-9, 'ConvertFrom', 'posixtime'); %Note how MATLAB defines posix time in seconds...

acc_rows = cellfun(@(x) strcmp(x, 'ACC'), sensors);
acc_values = values(acc_rows,:);
acc_timestamps_double = timestamps_double(acc_rows);
acc_timestamps = datetime(timestamps(acc_rows), 'Format', 'HH:mm');
%% 2. Select bounds (can be re-run or datatips deleted in figure) DON'T CLOSE if you want to keep the markers!!!
% Use datatips to select bounds between users
% Hold ALT to have multiple datatips

h = figure; hold all;
title(sprintf('User: %s, Date: %s, Label: %s, Flag: %s', user, recording_date, head, flag))
plot(acc_timestamps, acc_values(:,1), 'linewidth', 1)
plot(acc_timestamps, acc_values(:,2), 'linewidth', 1)
plot(acc_timestamps, acc_values(:,3), 'linewidth', 1)
box on;datacursormode on;
disp('Pick points')

dcm_obj = datacursormode(h);
set(dcm_obj,'DisplayStyle','datatip','SnapToDataVertex','off','Enable','on');

%% 3. Get cursor info
c_info = getCursorInfo(dcm_obj);
close all;

if isempty(c_info)
    error('no cursors in plot - redo step 2!')
end
%% 4. Plot selected intervals, and enter names (can be re-run if mistake in naming)
% Enter nothing to discard zone
close all;

cursor_dataindices = arrayfun(@(x) x.DataIndex, c_info);
cursor_dataindices = [1, sort(cursor_dataindices), length(acc_timestamps)];
window_starts = cursor_dataindices(1:(end-1));
true_window_starts = zeros(size(window_starts));
window_stops = cursor_dataindices(2:end);
true_window_stops = zeros(size(window_stops));
window_name = strings(length(window_starts),1);

for i=1:length(window_starts)
    h = figure;
    sp(1) = subplot(2,1,1); hold all;
    this_window_ix = window_starts(i):window_stops(i);
    plot(acc_timestamps, acc_values(:,1), 'b', 'linewidth', 1)
    plot(acc_timestamps, acc_values(:,2), 'r', 'linewidth', 1)
    plot(acc_timestamps, acc_values(:,3), 'g', 'linewidth', 1)
    
    plot(acc_timestamps(this_window_ix), acc_values(this_window_ix,1), 'b', 'linewidth', 3)
    plot(acc_timestamps(this_window_ix), acc_values(this_window_ix,2), 'r', 'linewidth', 3)
    plot(acc_timestamps(this_window_ix), acc_values(this_window_ix,3), 'y', 'linewidth', 3)
    axis tight
    
    sp(2) = subplot(2,1,2); hold all;
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
    window_name(i) = input('Enter name(or nothing to discard window): ', 's');
    if ~isempty(window_name(i))
        true_window_starts(i) = find(timestamps_double >= acc_timestamps_double(this_window_ix(1)),1,'first'); %first element in window
        true_window_stops(i) = find(timestamps_double < acc_timestamps_double(this_window_ix(end)),1,'last'); %last element not considered in window
    end
    close(h)
end

step4 = true;

%% 5. Generate folders and files (keeping the old folderstyle for compatibility)
for i=1:length(window_name)
    if (window_name(i) ~= "") %discarding unnamed windows
        path = fullfile(save_folder_root, 'sorted', window_name(i));
        mkdir(path);
        save_name = strcat(sprintf('%10.0f', timestamps_double(true_window_starts(i))), '_', watch_number, '.csv');
        save_file_path = fullfile(path,save_name);
        f = fopen(save_file_path, 'w');
        fprintf(f, sprintf('%s\n', head));
        for ii=true_window_starts(i):true_window_stops(i)
            line = sprintf('%s; %s; %f; %f; %f\n', timestamps_cells{ii}, sensors{ii}, ...
                   values(ii,1), values(ii,2), values(ii,3));
            fprintf(f, line);
        end
        eoftag = sprintf('%s\n', flag);
        fprintf(f, eoftag);
        fclose(f);
    end
end

clear step4