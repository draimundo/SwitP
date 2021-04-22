package com.wirdanie.switp;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.os.AsyncTask;
import android.util.Log;

import org.apache.commons.lang3.ArrayUtils;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.function.BiConsumer;

import static java.util.concurrent.Executors.newSingleThreadExecutor;

public class NeuralNetworkClient {
    private static final String TAG = "NeuralNetworkClient";
    private static final String MODEL_PATH = "SwitP_5agm.tflite";
//    private static final String LABEL_PATH = "SwitP_5labels.txt"; //TODO: elsewhere

    private static final int N_SAMPLES = 180; // TODO: infer this from model
    private static final int N_CHANNELS = 3 * 3;
    private static final int N_CLASSES = 5;

    private final Context context;
    private final List<String> labels = new ArrayList<>();
    private Interpreter tflite;
    private BiConsumer<Long, Float[]> retFun = null;

    ExecutorService singleThreadExecutor = null;

    public NeuralNetworkClient(Context context, BiConsumer<Long, Float[]> retFun) {
        Log.d(TAG, "Instanciated");
        this.context = context;
        this.loadModel();
        this.retFun = retFun;
        this.singleThreadExecutor = newSingleThreadExecutor();
    }


    /**
     * Actual classification method
     **/
    public void classify(final DataLoggingService.Window window) {
        ClassificationTask classTask = new ClassificationTask(window);
        if(!this.singleThreadExecutor.isShutdown())
            classTask.executeOnExecutor(this.singleThreadExecutor);
    }

    public void returnValues(Long timestamp, Float[] values) {
        this.retFun.accept(timestamp, values);
    }

    /**
     * Load TF Lite model.
     */
    private synchronized void loadModel() {
        try {
            ByteBuffer buffer = loadModelFile(this.context.getAssets());
            tflite = new Interpreter(buffer);
            Log.v(TAG, "TFLite model loaded.");
        } catch (IOException ex) {
            Log.e(TAG, ex.getMessage());
        }
    }


    // TODO select correct model depending on N_CHANNELS, which can be given by DataProcessingClient

    /**
     * Load TF Lite model from assets.
     */
    private static MappedByteBuffer loadModelFile(AssetManager assetManager) throws IOException {
        try (AssetFileDescriptor fileDescriptor = assetManager.openFd(MODEL_PATH);
             FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor())) {
            FileChannel fileChannel = inputStream.getChannel();
            long startOffset = fileDescriptor.getStartOffset();
            long declaredLength = fileDescriptor.getDeclaredLength();
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
        }
    }

    public static int[] getInputDimensions() {
        int[] ret = {N_SAMPLES, N_CHANNELS}; // TODO: infer this from model
        return ret;
    }

    public static int getOutputDimension() {
        int ret = N_CLASSES; // TODO: infer this from model
        return ret;
    }

    private class ClassificationTask extends AsyncTask<Void, Void, float[][]> {
        private DataLoggingService.Window window = null;


        private ClassificationTask(DataLoggingService.Window window) {
            this.window = window;
        }

        protected float[][] doInBackground(Void... none) {
            //Log.v(TAG, "Classifying " + this.window.endTime + " with TF Lite");
            float[][] output = new float[1][getOutputDimension()];

            long startTime = System.nanoTime();


            if(tflite != null) {
                tflite.run(this.window.values, output); //TODO not sure the same interpreter can run multiple times in parallel
                float estimatedTime = ((float) System.nanoTime() - (float) startTime) / 1000000000f;
                Log.v(TAG, "Estimated " + this.window.endTime + " inference time:" + estimatedTime);
                return output;
            } else {
                Log.d(TAG, "TFlite was closed");
                return null;
            }
        }

        @Override
        protected void onPostExecute(float[][] output) {
            if(output != null) {
                returnValues((long) (window.endTime - 3 * 1E9), ArrayUtils.toObject(output[0])); //estimate on center of window
            }
        }
    }

// TODO: elsewhere

//    /** Load label files **/
//    private synchronized void loadLabels() {
//        try {
//            loadLabelFile(this.context.getAssets());
//            Log.v(TAG, "Labels loaded.");
//        } catch (IOException ex) {
//            Log.e(TAG, ex.getMessage());
//        }
//    }
//
//    /** Load labels from assets. */
//    private void loadLabelFile(AssetManager assetManager) throws IOException {
//        try (InputStream ins = assetManager.open(LABEL_PATH);
//             BufferedReader reader = new BufferedReader(new InputStreamReader(ins))) {
//            // Each line in the label file is a label.
//            while (reader.ready()) {
//                labels.add(reader.readLine());
//            }
//        }
//    }

    /**
     * Free up resources as the client is no longer needed.
     */
    public synchronized void shutdown() {
        this.singleThreadExecutor.shutdown();
        tflite.close();
        tflite = null;
    }

    Interpreter getTflite() {
        return this.tflite;
    }

}
