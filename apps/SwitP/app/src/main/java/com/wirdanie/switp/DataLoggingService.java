package com.wirdanie.switp;

import android.annotation.SuppressLint;
import android.app.Service;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.AsyncTask;
import android.os.Binder;
import android.os.IBinder;
import android.os.PowerManager;
import android.util.Log;

import org.apache.commons.lang3.ArrayUtils;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.NavigableSet;
import java.util.NoSuchElementException;
import java.util.Queue;
import java.util.TimeZone;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.ConcurrentSkipListMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;
import java.util.function.BiConsumer;

import static java.util.concurrent.Executors.newSingleThreadExecutor;
import static java.util.concurrent.Executors.newSingleThreadScheduledExecutor;

public class DataLoggingService extends Service implements SensorEventListener {
    // Binder given to clients
    private final IBinder binder = new LocalBinder();

    public class LocalBinder extends Binder {
        DataLoggingService getService() {
            // Return this instance of LocalService so clients can call public methods
            return DataLoggingService.this;
        }
    }

    @Override
    public IBinder onBind(Intent intent) {
        initDataLoggingService(intent);
        return binder;
    }

    @Override
    public boolean onUnbind(Intent intent){
        return false;
    }

    private SensorManager mSensorManager;
    private PowerManager mPowerManager;
    private static PowerManager.WakeLock mWakeLock;
    private FileWriter mWriter;
    private boolean recordingFinished = false;
    private static long MAX_WAKELOCK_DURATION = 1000000 * 6; // 64 minutes
    @SuppressLint("SimpleDateFormat")
    private SimpleDateFormat sdf = new SimpleDateFormat("dd.MM.yyyy-HH:mm:ss");;

    private static final String TAG = "DataLoggingService";

    private ConcurrentSkipListMap<Long, Window> windowMap = new ConcurrentSkipListMap<Long, Window>();
    private ConcurrentSkipListMap<Long, Float[]> classifiedMap = new ConcurrentSkipListMap<Long, Float[]>();

    private NeuralNetworkClient neuralNetworkClient;
    private TurnDetectionClient.LapDetector lapDetector;

    private ExecutorService WindowThreadExecutor = null;
    private ExecutorService GarbageThreadExecutor = null;
    private ExecutorService WriterThreadExecutor = null;

    private ScheduledExecutorService WindowTimerExecutor = null;
    private ScheduledFuture<?> WindowTimerExecutorHandle = null;

    private ScheduledExecutorService DataReadingTimerExecutor = null;
    private ScheduledFuture<?> DataReadingTimerExecutorHandle = null;
    private BufferedReader simReader = null;

    private int N_CHANNELS;
    private int nLaps = 0;
    private int nStrokes = 0;
    private Long firstMeasTimestamp = 0L;
    private Long lastMeasTimestamp = 0L;
    private Long lastWindowTimestamp = 0L;

    // Modifiable parameters

    private static final int SAMPLES_PER_WINDOW = 180;
    private static final long SECONDS_PER_WINDOW = 6;
    private static final long NANOSECONDS_PER_SECOND = (long) 1E9;
    private static final long NANOSECONDS_PER_WINDOW = SECONDS_PER_WINDOW * NANOSECONDS_PER_SECOND;
    private static final long NANOSECOND_WINDOW_OVERLAP = NANOSECONDS_PER_SECOND;
    private static final long GARBAGEBUFFERTIMEns = NANOSECONDS_PER_SECOND;

    private static final int WindowTimerExecutionDelayms = 50; // interval at which the length of all sensor buffers is polled
    private static final int SimulationLineReadingDelayms = 1; // period between sample reading during simulation

    private boolean SIMULATE_DATA;
    private static boolean SAVE_ORIGINAL;
    private static boolean SAVE_RESAMPLED;
    private static boolean SAVE_CLASSIFIED;
    private static boolean SAVE_LAP;
    private static boolean SAVE_STROKE;
    private static boolean POOL_LENGTH; // false:25m, true 50m

    private static long minLapTimens;
    private static long maxLapTimens;

    // Stroke counting parameters
    private static final int sc_lag = 5;
    private static final float sc_dynthreshold = 2f;
    private static final float sc_statthreshold = 0.25f;
    private static final float sc_influence = 0.15f;


    public DataLoggingService() {
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        initDataLoggingService(intent);
        return super.onStartCommand(intent, flags, startId);
    }

    void initDataLoggingService(Intent intent){

        Log.d(TAG, "initDataLoggingService");

        keepCPUAlive(MAX_WAKELOCK_DURATION);

        sdf.setTimeZone(TimeZone.getTimeZone("GMT"));

        final SharedPreferences sharedPreferences = getSharedPreferences(getString(R.string.preference_file_key), MODE_PRIVATE);
        SIMULATE_DATA = sharedPreferences.getBoolean(getString(R.string.key_pref_simulate),false);
        SAVE_ORIGINAL = sharedPreferences.getBoolean(getString(R.string.key_pref_save_original),true);
        SAVE_RESAMPLED = sharedPreferences.getBoolean(getString(R.string.key_pref_save_resampled),true);
        SAVE_CLASSIFIED = sharedPreferences.getBoolean(getString(R.string.key_pref_save_classified),true);
        SAVE_LAP = sharedPreferences.getBoolean(getString(R.string.key_pref_save_lap),true);
        SAVE_STROKE = sharedPreferences.getBoolean(getString(R.string.key_pref_save_lap), true);
        POOL_LENGTH = sharedPreferences.getBoolean(getString(R.string.key_pref_pool_length), true);

        if(POOL_LENGTH){ //50m
            minLapTimens = 22*NANOSECONDS_PER_SECOND;
            maxLapTimens = 75*NANOSECONDS_PER_SECOND;
        } else { //25m
            minLapTimens = 11*NANOSECONDS_PER_SECOND;
            maxLapTimens = 38*NANOSECONDS_PER_SECOND;
        }


        this.WindowThreadExecutor = newSingleThreadExecutor();
        this.GarbageThreadExecutor = newSingleThreadExecutor();
        this.WriterThreadExecutor = newSingleThreadExecutor();
        this.neuralNetworkClient = new NeuralNetworkClient(getApplicationContext(), this::addClassified);
        this.lapDetector = new TurnDetectionClient().new LapDetector(minLapTimens, maxLapTimens, this::addLap);

        this.WindowTimerExecutor = newSingleThreadScheduledExecutor();
        this.WindowTimerExecutorHandle = this.WindowTimerExecutor.scheduleWithFixedDelay(this::WindowTimer, 500, WindowTimerExecutionDelayms, TimeUnit.MILLISECONDS);

        if (SIMULATE_DATA) {
            Log.d(TAG,"SIMULATING");
            this.simReader = new BufferedReader(new InputStreamReader(getApplicationContext().getResources().openRawResource(R.raw.test2)));
            this.DataReadingTimerExecutor = newSingleThreadScheduledExecutor();
            this.DataReadingTimerExecutorHandle = this.DataReadingTimerExecutor.scheduleWithFixedDelay(this::ReadData, 100, SimulationLineReadingDelayms, TimeUnit.MILLISECONDS);
        }

        String currentTime = sdf.format(Calendar.getInstance().getTime());

        try {
            mWriter = new FileWriter(new File(getStorageDir(),
                    System.currentTimeMillis() + ".csv"));
            mWriter.write(String.format("%s\n", currentTime));
        } catch (IOException e) {
            Log.e(TAG, e.getMessage());
        }

        GarbageCleanerTask.registerGarbage((ConcurrentSkipListMap) windowMap);
        GarbageCleanerTask.registerGarbage((ConcurrentSkipListMap) classifiedMap);

        registerListeners(intent);

        Log.v(TAG, "DataLoggingService started");

    }


    @SuppressLint("InvalidWakeLockTag")
    public void keepCPUAlive(long maxDuration) {
        mPowerManager = (PowerManager) getSystemService(Context.POWER_SERVICE);
        mWakeLock = mPowerManager.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "MyWakelockTag");
        mWakeLock.acquire(maxDuration);
    }

    private class SensorWrapper {
        int sensorType;
        String sensorString;
        ConcurrentSkipListMap<Long, Float[]> sensorMap;
        ConcurrentSkipListMap<Long, Float[]> outputMap;

        Queue<BiConsumer<Long, Float[]>> resampledListeners = new ConcurrentLinkedQueue<BiConsumer<Long, Float[]>>();

        int sensorChannels;
        Sensor sensor;
        int SamplingPeriodUs;
        FilteringClient.Resampler resampler;

        SensorWrapper(int sensorType, String sensorString, int sensorChannels, Sensor sensor, int SamplingPeriodUs, int upFactor, int downFactor, int origFreq) {
            this.sensorType = sensorType;
            this.sensorString = sensorString;
            this.sensorMap = new ConcurrentSkipListMap<Long, Float[]>();
            GarbageCleanerTask.registerGarbage((ConcurrentSkipListMap) this.sensorMap);

            this.sensorChannels = sensorChannels;
            this.sensor = sensor;
            this.SamplingPeriodUs = SamplingPeriodUs;

            this.outputMap = new ConcurrentSkipListMap<Long, Float[]>();
            GarbageCleanerTask.registerGarbage((ConcurrentSkipListMap) this.outputMap);

            this.resampler = new FilteringClient().new Resampler(upFactor, downFactor, origFreq, this.sensorChannels, this::addResampled);
        }

        void addMeasurement(long timestamp, Float[] values) {
            //this.sensorMap.put(timestamp, values);
            this.resampler.resample(timestamp, values);
            if (SAVE_ORIGINAL)
                this.writeToFile(timestamp, values, "");
        }

        void addResampled(long timestamp, Float[] values) {
            this.outputMap.put(timestamp, values);

            for(BiConsumer<Long, Float[]> addFun : this.resampledListeners){
                addFun.accept(timestamp,values);
            }

            if (SAVE_RESAMPLED)
                this.writeToFile(timestamp, values, "F");
        }

        void addResampledListener(BiConsumer<Long, Float[]> addFun){
            this.resampledListeners.add(addFun);
        }

        void writeToFile(long timestamp, Float[] values, String spec) {
            String tmp = null;
            if (values.length == 3) {
                tmp = String.format("%d; %s; %f; %f; %f\n", timestamp, this.sensorString + spec, values[0], values[1], values[2]);
            } else if (this.sensorChannels == 1) {
                tmp = String.format("%d; %s; %f\n", timestamp, this.sensorString + spec, values[0]);
            }
            writeToCSV(tmp);
        }
    }

    void addClassified(long timestamp, Float[] values) {
        this.classifiedMap.put(timestamp, values);
        this.lapDetector.addMeasurement(timestamp, values[0]);
        if (SAVE_CLASSIFIED) {
            StringBuilder tmp = new StringBuilder();
            tmp.append(String.format("%d; CLASS; ", timestamp));
            for (int i = 0; i < values.length - 1; i++) {
                tmp.append(String.format("%f;", values[i]));
            }
            tmp.append(String.format("%f\n", values[values.length - 1]));
            writeToCSV(tmp.toString());
        }
    }

    void addLap(long timestamp){
        this.nLaps++;
        if (SAVE_LAP){
            String tmp = String.format("%d; LAP\n", timestamp);
            writeToCSV(tmp);
        }
    }

    void newStroke(long timestamp){
        this.nStrokes++;
        if (SAVE_STROKE) {
            String tmp = String.format("%d; STROKE\n", timestamp);
            writeToCSV(tmp);
        }
    }

    void  WindowTimer() {
        // See if 1s elapsed
        Long smallestTimestamp = new Long(Long.MAX_VALUE);

        for (Map.Entry<Integer, SensorWrapper> sensor : usedSensors.entrySet()) {
            try {
                Long lastTS = sensor.getValue().outputMap.lastKey();
                smallestTimestamp = Math.min(lastTS, smallestTimestamp);
            } catch (NoSuchElementException e) {
                Log.d(TAG, "Empty map");
            }
        }

        if ((smallestTimestamp - lastWindowTimestamp) >= NANOSECOND_WINDOW_OVERLAP) {
            ArrayList<ConcurrentSkipListMap<Long, Float[]>> croppedAllOutputMaps = new ArrayList<ConcurrentSkipListMap<Long, Float[]>>();
            // See if all sensors have enough recordings older than the smallest timestamp, return if not
            for (Map.Entry<Integer, SensorWrapper> sensor : usedSensors.entrySet()) {
                // Log.d(TAG, sensor.getValue().sensorString + " has " + sensor.getValue().outputMap.headMap(smallestTimestamp, true).size());
                if (sensor.getValue().outputMap.headMap(smallestTimestamp, true).size() < SAMPLES_PER_WINDOW) {
                    return; //not enough samples on every channel
                } else {
                    croppedAllOutputMaps.add(sensor.getValue().outputMap); // build the array for the normalizing service
                }
            }
            //Log.d(TAG, "New window " + smallestTimestamp*1E-9);
            if(!WindowThreadExecutor.isShutdown())
                new WindowCreationTask(smallestTimestamp, croppedAllOutputMaps).executeOnExecutor(WindowThreadExecutor);
            if(!GarbageThreadExecutor.isShutdown())
                new GarbageCleanerTask(smallestTimestamp, GARBAGEBUFFERTIMEns, NANOSECONDS_PER_WINDOW).executeOnExecutor(GarbageThreadExecutor);
        }
    }

    private class WindowCreationTask extends AsyncTask<Void, Void, Window> {
        ArrayList<ConcurrentSkipListMap<Long, Float[]>> allOutputMaps = null;
        Long timestamp;

        WindowCreationTask(Long timestamp, ArrayList<ConcurrentSkipListMap<Long, Float[]>> allOutputMaps) {
            this.timestamp = timestamp;
            this.allOutputMaps = allOutputMaps;
        }

        protected Window doInBackground(Void... nothing) {
            int i_channel = 0;

            float[] sum = new float[N_CHANNELS];
            float[] ss = new float[N_CHANNELS];


            float[][][][] tmpvalues = new float[1][SAMPLES_PER_WINDOW][N_CHANNELS][1];

            for (ConcurrentSkipListMap<Long, Float[]> outputMap : allOutputMaps) {
                int i_meas = SAMPLES_PER_WINDOW - 1;
                for (Float[] sensorValues : outputMap.descendingMap().values()) { //take most recent values
                    for (int i_sensorChannels = 0; i_sensorChannels < sensorValues.length; i_sensorChannels++) { // copy values of every sensor channel in window table
                        tmpvalues[0][i_meas][i_sensorChannels + i_channel][0] = sensorValues[i_sensorChannels];
                        sum[i_sensorChannels + i_channel] += sensorValues[i_sensorChannels];
                        ss[i_sensorChannels + i_channel] += sensorValues[i_sensorChannels] * sensorValues[i_sensorChannels];
                    }
                    i_meas--;
                    if (i_meas < 0) { // window full for this channel
                        i_channel += sensorValues.length;
                        break;
                    }
                }
            }

            // Normalize data
            for (i_channel = 0; i_channel < N_CHANNELS; i_channel++) {
                float avg = (float) sum[i_channel] / ((float) SAMPLES_PER_WINDOW);
                float std_dev = (float) Math.sqrt(ss[i_channel] / ((float) SAMPLES_PER_WINDOW) - avg * avg);

                for (int i_sample = 0; i_sample < SAMPLES_PER_WINDOW; i_sample++) {
                    tmpvalues[0][i_sample][i_channel][0] = (tmpvalues[0][i_sample][i_channel][0] - avg) / std_dev;
                }
            }
            return new Window(this.timestamp, tmpvalues);
        }

        protected void onPostExecute(Window window) {
            //windowMap.put(window.endTime, window);
            neuralNetworkClient.classify(window);
            lastWindowTimestamp = window.endTime;
        }
    }

    private static class GarbageCleanerTask extends AsyncTask<Void, Void, Void> {
        private static List<ConcurrentSkipListMap<Long, Object>> garbageList = new ArrayList<ConcurrentSkipListMap<Long, Object>>();
        long WINDOWLENGTHns = 0;
        long BUFFERTIMEns = 0;
        long lastTimestamp = 0;

        GarbageCleanerTask(long lastTimestamp, long BUFFERTIMEns, long WINDOWLENGTHns) {
            this.WINDOWLENGTHns = WINDOWLENGTHns;
            this.BUFFERTIMEns = BUFFERTIMEns;
            this.lastTimestamp = lastTimestamp;
        }

        static void registerGarbage(ConcurrentSkipListMap<Long, Object> map) {
            garbageList.add(map);
        }

        protected Void doInBackground(Void... nothing) {
            long deleteBefore = this.lastTimestamp - this.BUFFERTIMEns - this.WINDOWLENGTHns;

            for (ConcurrentSkipListMap<Long, Object> map : garbageList) {
                NavigableSet<Long> oldMap = map.headMap(deleteBefore).keySet();
                oldMap.clear();
            }
            return null;
        }
    }

    private void ReadData() {
        String line = "";
        try {
            if ((line = this.simReader.readLine()) != null) {
                String[] sensorLine = line.split("; ");
                if (sensorLine.length == 5) {
                    SensorWrapper sensor = usedSensorsString.get(sensorLine[1]);
                    if (sensor != null) { // ignore not used Sensors
                        Float[] values = new Float[sensorLine.length - 2];
                        for (int i = 2; i < sensorLine.length; i++) {
                            values[i - 2] = Float.parseFloat(sensorLine[i]);
                        }
                        onSensorChanged(sensor, Long.parseLong(sensorLine[0]), values);
                    }
                } else {
                    Log.d(TAG, "Non standard format (maybe header/footer?)" + line);
                }
            }
        } catch (IOException e) {
            Log.e(TAG, e.getMessage());
        }
    }

    private void writeToCSV(String str){
        if(!WriterThreadExecutor.isShutdown())
            new WritingTask(str).executeOnExecutor(this.WriterThreadExecutor);
    }

    private class WritingTask extends AsyncTask<Void, Void, Void> {
        String str;

        WritingTask(String str) {
            this.str = str;
        }

        @Override
        protected Void doInBackground(Void... voids) {
            try {
                mWriter.write(this.str);
            } catch (IOException e) {
                e.printStackTrace();
            }
            return null;
        }
    }

    private SensorWrapper sensorAccelerometer;
    private SensorWrapper sensorGyroscope;
    private SensorWrapper sensorMagnetometer;
    private SensorWrapper sensorLight;
    private SensorWrapper sensorTemperature;
    private SensorWrapper sensorPressure;

    private ArrayList<SensorWrapper> allSensors = new ArrayList<SensorWrapper>();

    private ConcurrentHashMap<Integer, SensorWrapper> usedSensors = new ConcurrentHashMap<Integer, SensorWrapper>();
    private HashMap<String, SensorWrapper> usedSensorsString = new HashMap<String, SensorWrapper>();

    public void registerListeners(Intent intent) {
        mSensorManager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);
        sensorAccelerometer = new SensorWrapper(Sensor.TYPE_ACCELEROMETER, "ACC", 3, mSensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER), 20000, 3, 5, 52);
        sensorGyroscope = new SensorWrapper(Sensor.TYPE_GYROSCOPE, "GYRO", 3, mSensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE), 20000, 3, 5, 52);
        sensorMagnetometer = new SensorWrapper(Sensor.TYPE_MAGNETIC_FIELD, "MAG", 3, mSensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD), 20000, 3, 5, 50);
        //sensorLight = new SensorWrapper(Sensor.TYPE_LIGHT, "LIGHT", new TreeMap<Long, Float[]>(), 1, mSensorManager.getDefaultSensor(Sensor.TYPE_LIGHT));
        //sensorTemperature = new SensorWrapper(Sensor.TYPE_AMBIENT_TEMPERATURE, "TEMP", new TreeMap<Long, Float[]>(), 1, mSensorManager.getDefaultSensor(Sensor.TYPE_AMBIENT_TEMPERATURE));
        //sensorPressure = new SensorWrapper(Sensor.TYPE_PRESSURE, "PRESS", new TreeMap<Long, Float[]>(), 1, mSensorManager.getDefaultSensor(Sensor.TYPE_PRESSURE));


        // Also defines the order of measurements in the window
        allSensors.add(sensorAccelerometer);
        allSensors.add(sensorGyroscope);
        allSensors.add(sensorMagnetometer);
        allSensors.add(sensorLight);
        allSensors.add(sensorTemperature);
        allSensors.add(sensorPressure);

        // Save the active sensors in usedSensors, register listeners
        for (SensorWrapper sensor : allSensors) {
            if (sensor != null) {
                usedSensors.put(sensor.sensorType, sensor);
                usedSensorsString.put(sensor.sensorString, sensor);
                N_CHANNELS += sensor.sensorChannels;
                if (!SIMULATE_DATA && sensor.sensor != null) {
                    mSensorManager.registerListener(this, sensor.sensor, sensor.SamplingPeriodUs);
                }
            }
        }

        registerStrokeCounting();
    }

    private void registerStrokeCounting(){
        final StrokeCountingClient strokeCountingClient = new StrokeCountingClient(sc_lag, sc_dynthreshold, sc_statthreshold, sc_influence, this::newStroke);
        final FilteringClient.Resampler resampler = new FilteringClient().new Resampler(1, 10, 300, 3, strokeCountingClient::addSample);
        usedSensorsString.get("ACC").addResampledListener(resampler::resample);
    }

    // Original method, called by the android OS
    public void onSensorChanged(SensorEvent event) {
        onSensorChanged(usedSensors.get(event.sensor.getType()), event.timestamp, ArrayUtils.toObject(event.values));
    }

    // Wrapper method, to be able to simulate
    public void onSensorChanged(SensorWrapper sensor, long timestamp, Float[] values) {
        sensor.addMeasurement(timestamp, values);
        this.lastMeasTimestamp = timestamp;
        if(this.firstMeasTimestamp == 0L){
            this.firstMeasTimestamp = timestamp;
        }
    }

    private String getStorageDir() {
        return this.getExternalFilesDir(null).getAbsolutePath();
    }

    @Override
    public void onDestroy() {
        if (DataReadingTimerExecutorHandle != null) {
            DataReadingTimerExecutorHandle.cancel(true);
        }

        this.neuralNetworkClient.shutdown();
        this.GarbageThreadExecutor.shutdown();
        this.WriterThreadExecutor.shutdown();

        String currentTime = sdf.format(Calendar.getInstance().getTime());
        try {
            mWriter.write(String.format("%s; %s\n", "Finished:", currentTime));
            mWriter.close();
        } catch (IOException e) {
            Log.e(TAG, e.getMessage());
        }

        mSensorManager.unregisterListener(this);

        mWakeLock.release();
        super.onDestroy();
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {
    }

    /**
     * Formatted/Normalized window to be fed to the neural network
     **/
    public class Window {
        Long endTime;
        float[][][][] values;

        Window(Long endTime, float[][][][] values) {
            this.endTime = endTime;
            this.values = values;
        }
    }

    public int getLaps(){
        return this.nLaps;
    }

    public int getStrokes(){
        return this.nStrokes;
    }

    public long getRunningTimens(){
        return this.lastMeasTimestamp - this.firstMeasTimestamp;
    }

}
