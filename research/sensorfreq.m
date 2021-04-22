ts_acc = timestamp(find(type == "ACC"));
ts_mag = timestamp(find(type == "MAG"));
ts_gyro = timestamp(find(type == "GYRO"));
% ts_light = timestamp(find(type == "LIGHT"));
% ts_press = timestamp(find(type == "PRESS"));
% ts_temp = timestamp(find(type == "TEMP"));

ts_diff_acc = ts_acc(2:end)-ts_acc(1:end-1);
ts_diff_mag = ts_mag(2:end)-ts_mag(1:end-1);
ts_diff_gyro = ts_gyro(2:end)-ts_gyro(1:end-1);

% ts_diff_light = ts_light(2:end)-ts_light(1:end-1);
% ts_diff_press = ts_press(2:end)-ts_press(1:end-1);
% ts_diff_temp = ts_temp(2:end)-ts_temp(1:end-1);



figure;plot(ts_diff_acc(3:end),'x');title("ACC")
figure;plot(ts_diff_mag(3:end),'x');title("MAG")
figure;plot(ts_diff_gyro(3:end),'x');title("GYRO")
% figure;plot(ts_diff_light(3:end),'x');title("LIGHT")
% figure;plot(ts_diff_press(3:end),'x');title("PRESS")
% figure;plot(ts_diff_temp(3:end),'x');title("TEMP")

avg_acc = 1E9/mean(ts_diff_acc);
avg_mag = 1E9/mean(ts_diff_mag);
avg_gyro = 1E9/mean(ts_diff_gyro);
% avg_light = 1E9/mean(ts_diff_light);
% avg_press = 1E9/mean(ts_diff_press);
% avg_temp = 1E9/mean(ts_diff_temp);
