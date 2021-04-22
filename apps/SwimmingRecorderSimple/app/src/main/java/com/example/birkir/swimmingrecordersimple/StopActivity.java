package com.example.birkir.swimmingrecordersimple;

import android.content.Intent;
import android.os.Bundle;
import android.support.wearable.activity.WearableActivity;
import android.view.GestureDetector;
import android.view.MotionEvent;
import android.view.View;
import android.widget.TextView;

import org.w3c.dom.Text;

public class StopActivity extends WearableActivity {

    private TextView mTvStopTop;
    private TextView mTvStopBottom;
    private TextView mTvStopping;
    private GestureDetector mSwipe;
    private static final int SWIPE_MIN_DISTANCE = 100;
    private static final int SWIPE_MAX_TIME_DIFFERENCE = 4000; // 1 second
    private static final int SWIPE_MIN_COUNT = 10;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_stop);

        // Enables Always-on
        setAmbientEnabled();

        mTvStopping = (TextView) findViewById(R.id.tvStopping);
        mTvStopTop = (TextView) findViewById(R.id.tvStopTop);
        mTvStopBottom = (TextView) findViewById(R.id.tvStopBottom);
        mSwipe = new GestureDetector(this, new StopSwipeDetector());
    }

    private class StopSwipeDetector extends GestureDetector.SimpleOnGestureListener {
        private boolean isLoading = false;
        private long lastSwipe = 0;
        private int swipeCount = 0;
        @Override
        public boolean onScroll(MotionEvent e1, MotionEvent e2, float distanceX, float distanceY) {

            double x1 = e1.getX();
            double x2 = e2.getX();
            double y1 = e1.getY();
            double y2 = e2.getY();
            double d = Math.sqrt(Math.pow(y2-y1,2) + Math.pow(x2-x1,2));

            if (d >= SWIPE_MIN_DISTANCE) {
                long dt = e1.getEventTime() - lastSwipe;
                if (dt > 0) {
                    lastSwipe = e1.getEventTime();
                    if (dt <= SWIPE_MAX_TIME_DIFFERENCE) {
                        swipeCount = swipeCount + 1;
                    }
                    else {
                        swipeCount = 1;
                    }
                }
                if (swipeCount == SWIPE_MIN_COUNT && !isLoading) {
                    isLoading = true;
                    mTvStopping.setVisibility(View.VISIBLE);
                    mTvStopTop.setVisibility(View.INVISIBLE);
                    mTvStopBottom.setVisibility(View.INVISIBLE);
                    Intent stopRecIntent = new Intent(); // Maybe this should be different
                    stopRecIntent.setAction("com.example.birkir.swimmingrecordersimple.STOP_RECORDING");
                    sendBroadcast(stopRecIntent);

                    Intent resultsIntent = new Intent(StopActivity.this, ResultsActivity.class);
                    startActivity(resultsIntent);
                    finish();
                    return true;
                }
            }
//            if (d >= SWIPE_MIN_DISTANCE) {
//                long dt = e1.getEventTime() - lastSwipe;
//                if (dt <= SWIPE_MAX_TIME_DIFFERENCE && dt > 0 && !isLoading) {
//                    mTvStopping.setVisibility(View.VISIBLE);
//                    mTvStopTop.setVisibility(View.INVISIBLE);
//                    mTvStopBottom.setVisibility(View.INVISIBLE);
//                    isLoading = true;
//                    Intent stopRecIntent = new Intent(); // Maybe this should be different
//                    stopRecIntent.setAction("com.example.birkir.swimmingrecordersimple.STOP_RECORDING");
//                    sendBroadcast(stopRecIntent);
//
//                    Intent resultsIntent = new Intent(StopActivity.this, ResultsActivity.class);
//                    startActivity(resultsIntent);
//                    finish();
//                    return true;
//                }
//                lastSwipe = e1.getEventTime();
//            }
            return false;
        }
    }

    // Somehow not including dispatchTouchEvent gets rid of the delay. But this method is needed in
    // StartActivity.

    @Override
    public boolean onTouchEvent(MotionEvent event) {
        return mSwipe.onTouchEvent(event);
    }
}
