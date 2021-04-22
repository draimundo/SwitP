package com.wirdanie.switp;

import android.annotation.SuppressLint;
import android.content.ComponentName;
import android.content.Context;
import android.content.Intent;
import android.content.ServiceConnection;
import android.os.Bundle;
import android.os.IBinder;

import androidx.appcompat.app.AppCompatActivity;

import android.util.Log;
import android.view.MotionEvent;
import android.view.View;
import android.view.WindowManager;
import android.widget.TextView;

import androidx.wear.ambient.AmbientModeSupport;
import androidx.wear.widget.CircularProgressLayout;

import org.w3c.dom.Text;

import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.TimeZone;
import java.util.Timer;
import java.util.TimerTask;

public class RecordingActivity extends AppCompatActivity implements
        AmbientModeSupport.AmbientCallbackProvider {

    private final int updateRatems = 250;

    private TextView mTvTime;
    private TextView mTvLaps;
    private TextView mTvStrokes;

    private CircularProgressLayout mStopProgress;
    private static final String TAG = "RecordingActivity";

    DataLoggingService dataLoggingService;
    boolean mBound = false;

    @SuppressLint("SimpleDateFormat")
    SimpleDateFormat sdf = new SimpleDateFormat("HH:mm:ss.SS");

    @SuppressLint("ClickableViewAccessibility")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_recording);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);


        Intent startIntent = new Intent(this, DataLoggingService.class);
        bindService(startIntent, connection, Context.BIND_AUTO_CREATE);

        sdf.setTimeZone(TimeZone.getTimeZone("GMT"));

        mTvTime = (TextView) findViewById(R.id.timerView);
        mTvLaps = (TextView) findViewById(R.id.lapView);
        mTvStrokes = (TextView) findViewById(R.id.strokeView);

        mStopProgress = findViewById(R.id.stop_recording);
        mStopProgress.setOnTimerFinishedListener(new CircularProgressLayout.OnTimerFinishedListener() {
            @Override
            public void onTimerFinished(CircularProgressLayout layout) {
                mStopProgress.stopTimer();

                createDialog("Stop recording?");
            }
        });
        mStopProgress.setOnTouchListener(new View.OnTouchListener() {
            @Override
            public boolean onTouch(View view, MotionEvent motionEvent) {
                int action = motionEvent.getAction();
                Log.d(TAG, "Action:" + action);
                switch (action) {
                    case MotionEvent.ACTION_DOWN:
                    case MotionEvent.ACTION_MOVE:
                        mStopProgress.setTotalTime(2000);
                        mStopProgress.startTimer();
                        break;
                    default:
                        mStopProgress.stopTimer();
                }
                return true;
            }
        });
        updateFieldTimer.schedule(updateFieldTask, 0, updateRatems);
    }

    @Override
    protected void onDestroy() {
        unbindService(connection);
        dataLoggingService.stopSelf();
        mBound = false;
        super.onDestroy();
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        // Collect data from the intent and use it
        if (resultCode == RESULT_OK && requestCode == 1) {
            finish();
        }
    }

    private ServiceConnection connection = new ServiceConnection() {
        @Override
        public void onServiceConnected(ComponentName className, IBinder service) {
            // We've bound to LocalService, cast the IBinder and get LocalService instance
            DataLoggingService.LocalBinder binder = (DataLoggingService.LocalBinder) service;
            dataLoggingService = binder.getService();
            mBound = true;
        }

        @Override
        public void onServiceDisconnected(ComponentName arg0) {
            mBound = false;
        }
    };

    final Runnable updateFieldRunnable = new Runnable() {
        public void run() {
            if (mBound) {
                String nLaps = String.valueOf(dataLoggingService.getLaps());
                mTvLaps.setText(nLaps);

                String HMSms = sdf.format(new Date(Math.round((double) dataLoggingService.getRunningTimens() / 1E6)));
                mTvTime.setText(HMSms);

                String nStrokes = String.valueOf(dataLoggingService.getStrokes());
                mTvStrokes.setText(nStrokes);
            }
        }
    };

    private void createDialog(String DialogText) {
        Intent intent = new Intent(this, ConfirmDialog.class);
        intent.putExtra("Dialog_Text", DialogText);
        int requestCode = 1;
        startActivityForResult(intent, requestCode);
    }

    TimerTask updateFieldTask = new TimerTask() {
        @Override
        public void run() {
            runOnUiThread(updateFieldRunnable);
        }
    };

    Timer updateFieldTimer = new Timer();

    @Override
    public AmbientModeSupport.AmbientCallback getAmbientCallback() {
        return new MyAmbientCallback();
    }

    private class MyAmbientCallback extends AmbientModeSupport.AmbientCallback {
        /**
         * Prepares the UI for ambient mode.
         */
        @Override
        public void onEnterAmbient(Bundle ambientDetails) {
            super.onEnterAmbient(ambientDetails);
            Log.d(TAG, "onEnterAmbient() " + ambientDetails);
        }

        /**
         * Restores the UI to active (non-ambient) mode.
         */
        @Override
        public void onExitAmbient() {
            super.onExitAmbient();
            Log.d(TAG, "onExitAmbient()");
        }
    }


}
