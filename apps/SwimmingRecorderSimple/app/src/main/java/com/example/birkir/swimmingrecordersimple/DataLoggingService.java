package com.example.birkir.swimmingrecordersimple;

import android.app.Service;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.IBinder;
import android.os.PowerManager;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Calendar;

public class DataLoggingService extends Service implements SensorEventListener {
    private SensorManager mSensorManager;
    private PowerManager mPowerManager;
    private static PowerManager.WakeLock mWakeLock;
    private BroadcastReceiver mMessageReceiver;
    private FileWriter mWriter;
    private boolean recordingFinished = false;
    private static long MAX_WAKELOCK_DURATION = 1000000*6; // 64 minutes
    private SimpleDateFormat sdfStart;
    private SimpleDateFormat sdfStop;

    public DataLoggingService() {
    }

    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        String style = intent.getStringExtra("style");
        keepCPUAlive(MAX_WAKELOCK_DURATION);
        registerListeners();
        registerReceivers();
        sdfStart = new SimpleDateFormat("dd.MM.yyyy-HH:mm:ss");
        String currentTime = sdfStart.format(Calendar.getInstance().getTime());
        try {
            mWriter = new FileWriter(new File(getStorageDir(), style + "_" +
                    System.currentTimeMillis() + ".csv"));
            mWriter.write(String.format("%s; %s\n", style, currentTime));
        }
        catch (IOException e) {
            e.printStackTrace();
        }

        return super.onStartCommand(intent, flags, startId);
    }

    public void keepCPUAlive(long maxDuration) {
        mPowerManager = (PowerManager) getSystemService(Context.POWER_SERVICE);
        mWakeLock = mPowerManager.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "MyWakelockTag");
        mWakeLock.acquire(maxDuration);
    }

    public void registerListeners() {
        Sensor sensorAccelerometer;
        Sensor sensorGyroscope;
        Sensor sensorMagnetometer;
        Sensor sensorLight;
        Sensor sensorTemperature;
        Sensor sensorPressure;

        mSensorManager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);

        sensorAccelerometer = mSensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        sensorGyroscope = mSensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE);
        sensorMagnetometer = mSensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD);
        sensorLight = mSensorManager.getDefaultSensor(Sensor.TYPE_LIGHT);
        sensorTemperature = mSensorManager.getDefaultSensor(Sensor.TYPE_AMBIENT_TEMPERATURE);
        sensorPressure = mSensorManager.getDefaultSensor(Sensor.TYPE_PRESSURE);

        if (sensorAccelerometer != null) {
            mSensorManager.registerListener(this, sensorAccelerometer,
                    SensorManager.SENSOR_DELAY_FASTEST);
        }
        if (sensorGyroscope != null) {
            mSensorManager.registerListener(this, sensorGyroscope,
                    SensorManager.SENSOR_DELAY_FASTEST);
        }
        if (sensorMagnetometer != null) {
            mSensorManager.registerListener(this, sensorMagnetometer,
                    SensorManager.SENSOR_DELAY_FASTEST);
        }
        if (sensorLight != null) {
            mSensorManager.registerListener(this, sensorLight,
                    SensorManager.SENSOR_DELAY_FASTEST);
        }
        if (sensorTemperature != null) {
            mSensorManager.registerListener(this, sensorTemperature,
                    SensorManager.SENSOR_DELAY_FASTEST);
        }
        if (sensorPressure != null) {
            mSensorManager.registerListener(this, sensorPressure,
                    SensorManager.SENSOR_DELAY_FASTEST);
        }
    }

    public void registerReceivers() {
        mMessageReceiver = new MyReceiver();
        IntentFilter intentFilter = new IntentFilter();
        intentFilter.addAction("com.example.birkir.swimmingrecordersimple.STOP_RECORDING");
        intentFilter.addAction("com.example.birkir.swimmingrecordersimple.FINISH_RECORDING");
        registerReceiver(mMessageReceiver, intentFilter);
    }

    private class MyReceiver extends BroadcastReceiver {

        @Override
        public void onReceive(Context context, Intent intent) {
            switch (intent.getAction()) {
                case "com.example.birkir.swimmingrecordersimple.STOP_RECORDING":
                    unregisterListeners();
                    break;
                case "com.example.birkir.swimmingrecordersimple.FINISH_RECORDING":
                    recordingFinished = true;
                    String fileTag = intent.getStringExtra("fileTag");
                    sdfStop = new SimpleDateFormat("dd.MM.yyyy-HH:mm:ss");
                    String currentTime = sdfStop.format(Calendar.getInstance().getTime());
                    try {
                        mWriter.write(String.format("%s; %s\n", fileTag, currentTime));
                        mWriter.close();
                    }
                    catch (IOException e) {
                        e.printStackTrace();
                    }
                    break;
            }
        }
    }

    public void onSensorChanged(SensorEvent event) {
        try {
            switch (event.sensor.getType()) {
                case Sensor.TYPE_ACCELEROMETER:
                    mWriter.write(String.format("%d; ACC; %f; %f; %f\n",
                            event.timestamp, event.values[0], event.values[1], event.values[2]));
                    break;
                case Sensor.TYPE_GYROSCOPE:
                    mWriter.write(String.format("%d; GYRO; %f; %f; %f\n",
                            event.timestamp, event.values[0], event.values[1], event.values[2]));
                    break;
                case Sensor.TYPE_MAGNETIC_FIELD:
                    mWriter.write(String.format("%d; MAG; %f; %f; %f\n",
                            event.timestamp, event.values[0], event.values[1], event.values[2]));
                    break;
                case Sensor.TYPE_LIGHT:
                    mWriter.write(String.format("%d; LIGHT; %f; %f; %f\n",
                            event.timestamp, event.values[0], 0.f, 0.f));
                    break;
                case Sensor.TYPE_AMBIENT_TEMPERATURE:
                    mWriter.write(String.format("%d; TEMP; %f; %f; %f\n",
                            event.timestamp, event.values[0], 0.f, 0.f));
                    break;
                case Sensor.TYPE_PRESSURE:
                    mWriter.write(String.format("%d; PRESS; %f; %f; %f\n",
                            event.timestamp, event.values[0], 0.f, 0.f));
                    break;
            }
        }
        catch (IOException e) {
            e.printStackTrace();
        }
    }

    private String getStorageDir() {
        return this.getExternalFilesDir(null).getAbsolutePath();
    }

    public void unregisterListeners() {
        mSensorManager.unregisterListener(this);
    }

    @Override
    public void onDestroy() {
        if (!recordingFinished) {
            try {
                String fileTag = "Destroyed";
                mWriter.write(String.format("%s\n", fileTag));
                mWriter.close();
            }
            catch (IOException e) {
                e.printStackTrace();
            }
        }
        mSensorManager.unregisterListener(this);
        mWakeLock.release();
        super.onDestroy();
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {

    }
}
