package com.example.birkir.swimmingrecordersimple;

import android.content.Intent;
import android.graphics.PorterDuff;
import android.os.Bundle;
import android.support.wearable.activity.WearableActivity;
import android.view.GestureDetector;
import android.view.MotionEvent;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

public class ResultsActivity extends WearableActivity {

    private Button mBtnSuccess;
    private Button mBtnFailure;
    private TextView mTvResults;
    private GestureDetector mSwipeDetector;
    double mSwipeDistance = 0;
    private static final int SWIPE_MAX_DISTANCE = 20;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_results);

        // Enables Always-on
        setAmbientEnabled();

        mSwipeDetector = new GestureDetector(this, new AccidentalSwipeDetector());

        mBtnSuccess = (Button) findViewById(R.id.btnSuccess);
        mBtnFailure = (Button) findViewById(R.id.btnFailure);
        mTvResults = (TextView) findViewById(R.id.tvResults);

        mBtnSuccess.getBackground().setColorFilter(0xFF00FF00, PorterDuff.Mode.MULTIPLY);
        mBtnFailure.getBackground().setColorFilter(0xFFFF0000, PorterDuff.Mode.MULTIPLY);

        mBtnSuccess.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                finishRecording("Success");
            }
        });
        mBtnFailure.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                finishRecording("Failure");
            }
        });
    }

    private void finishRecording(String fileTag) {
        if (mSwipeDistance <= SWIPE_MAX_DISTANCE) {
            mTvResults.setVisibility(View.VISIBLE);
            mBtnSuccess.setVisibility(View.INVISIBLE);
            mBtnFailure.setVisibility(View.INVISIBLE);
            Intent writeDataIntent = new Intent();
            writeDataIntent.setAction("com.example.birkir.swimmingrecordersimple.FINISH_RECORDING");
            writeDataIntent.putExtra("fileTag", fileTag);
            sendBroadcast(writeDataIntent);
            finish();
        }
        mSwipeDistance = 0;
    }

    private class AccidentalSwipeDetector extends GestureDetector.SimpleOnGestureListener {
        @Override
        public boolean onScroll(MotionEvent e1, MotionEvent e2, float distanceX, float distanceY) {
            double d = Math.sqrt(Math.pow(e2.getX() - e1.getX(), 2) + Math.pow(e2.getY() - e1.getY(), 2));
            if (d>SWIPE_MAX_DISTANCE) {
                mSwipeDistance = d;
            }
            return false;
        }
    }

    @Override
    public boolean dispatchTouchEvent(MotionEvent event) {
        // TouchEvent dispatcher.
        if (mSwipeDetector != null) {
            if (mSwipeDetector.onTouchEvent(event))
                return true;
        }
        return super.dispatchTouchEvent(event);
    }

    @Override
    public boolean onTouchEvent(MotionEvent event) {
        return mSwipeDetector.onTouchEvent(event);
    }

    @Override
    protected void onDestroy() {
        stopService(new Intent(ResultsActivity.this,
                DataLoggingService.class));
        super.onDestroy();
    }
}
