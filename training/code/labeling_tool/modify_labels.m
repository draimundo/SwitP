clear all
close all

user = 'unknownman';
recording_name = 'Freestyle_1527504541667.csv';

% Search for the full path to the recording
path_to_labeled = '..\..\data\labeled';
save_path = '..\..\data\re-labeled';
recording_dates = dir(path_to_labeled);
recording_dates = recording_dates(3:end);
recording_path = '';
for i=1:length(recording_dates)
    users_in_date = dir(fullfile(path_to_labeled, recording_dates(i).name));
    for ii=1:length(users_in_date)
        if strcmpi(user, users_in_date(ii).name)
            date_user_recordings = dir(fullfile(path_to_labeled, recording_dates(i).name, user));
            for iii=1:length(date_user_recordings)
                if strcmpi(date_user_recordings(iii).name, recording_name)
                    recording_date = recording_dates(i).name;
                    recording_path = fullfile(path_to_labeled, recording_dates(i).name, user, recording_name);
                end
            end
        end
    end
end
if isempty(recording_path)
    error('Recording doesnt exist')
end
disp(recording_path)

filestr = fileread(recording_path);
filebyline = regexp(filestr, '\n', 'split');
filebyfield = regexp(filebyline, '; ', 'split');
header = filebyfield(1);
footer = filebyfield(end);
data = filebyfield(2:end-1);
timestamps = cellfun(@(x) str2num(x{1}), data);
labels = cellfun(@(x) str2num(x{6}), data);
acc_rows = cellfun(@(x) strcmp(x{2}, 'ACC'), data);
acc_timestamps = cellfun(@(x) str2num(x{1}), data(acc_rows));
acc_0 = cellfun(@(x) str2num(x{3}), data(acc_rows));
acc_1 = cellfun(@(x) str2num(x{4}), data(acc_rows));
acc_2 = cellfun(@(x) str2num(x{5}), data(acc_rows));
acc_labels = cellfun(@(x) str2num(x{6}), data(acc_rows));

h = figure; hold all;
sp(1) = subplot(2,1,1); hold all
plot(acc_timestamps, acc_0)
plot(acc_timestamps, acc_1)
plot(acc_timestamps, acc_2)
sp(2) = subplot(2,1,2);
plot(acc_timestamps, acc_labels, 'linewidth', 2)
yticks([-1 0 1 2 3 4 5 6])
yticklabels({'unknown', 'null', 'freestyle', 'breaststroke', 'backstroke', 'butterfly', 'turn', 'kick'})
linkaxes(sp, 'x')
box on;
disp('Pick points')
uiwait(h);

if ~exist('cursor_info', 'var')
    error('no cursors in plot')
end

%%
cursor_timestamps = sort(arrayfun(@(x) x.Position(1), cursor_info));
window_starts = cursor_timestamps(1:2:end);
window_stops = cursor_timestamps(2:2:end);
window_label = zeros(length(window_starts),1);
acc_labels_modified = acc_labels;
for i=1:length(window_starts)
    h = figure; 
    sp(1) = subplot(2,1,1); hold all
    this_window_ix = find(acc_timestamps>=window_starts(i) & acc_timestamps<=window_stops(i));
    plot(acc_timestamps, acc_0, 'b', 'linewidth', 1)
    plot(acc_timestamps, acc_1, 'r', 'linewidth', 1)
    plot(acc_timestamps, acc_2, 'g', 'linewidth', 1)
    plot(acc_timestamps(this_window_ix), acc_0(this_window_ix), 'b', 'linewidth', 3)
    plot(acc_timestamps(this_window_ix), acc_1(this_window_ix), 'r', 'linewidth', 3)
    plot(acc_timestamps(this_window_ix), acc_2(this_window_ix), 'y', 'linewidth', 3)
    sp(2) = subplot(2,1,2); hold all;
    plot(acc_timestamps, acc_labels, 'linewidth', 2)
    window_label(i) = input('Enter label: ');
    close(h)
    acc_labels_modified(this_window_ix) = window_label(i);
end

figure; hold all;
sp(1) = subplot(2,1,1); hold all
plot(acc_0)
plot(acc_1)
plot(acc_2)
sp(2) = subplot(2,1,2); hold all
plot(acc_labels, 'linewidth', 2)
plot(acc_labels_modified, 'linewidth', 2)
yticks([-1 0 1 2 3 4 5 6])
yticklabels({'unknown', 'null', 'freestyle', 'breaststroke', 'backstroke', 'butterfly', 'turn', 'kick'})
linkaxes(sp, 'x')
box on;

labels_modified = labels;
for i=1:length(window_label)
    ix_change = find(timestamps>=window_starts(i) & timestamps<=window_stops(i));
    labels_modified(ix_change) = window_label(i);
end

% Write back to file
mkdir(fullfile(save_path, recording_date, user))
save_file_path = fullfile(save_path, recording_date, user, recording_name);
f = fopen(save_file_path, 'w');
line = header{1}{1};
for i=2:length(header{1})
    line = sprintf('%s; %s', line, header{1}{i});
end
line = sprintf('%s\n', line);
fprintf(f, line);
for i=1:length(data)
    line = sprintf('%s; %s; %s; %s; %s; %d\n', data{i}{1}, data{i}{2}, ...
                   data{i}{3}, data{i}{4}, data{i}{5}, labels_modified(i));
    fprintf(f, line);
end
line = footer{1}{1};
for i=2:length(footer{1})
    line = sprintf('%s; %s', line, footer{1}{i});
end
line = sprintf('%s\n', line);
fprintf(f, line);
fclose(f);

