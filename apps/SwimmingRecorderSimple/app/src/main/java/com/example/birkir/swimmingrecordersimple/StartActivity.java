package com.example.birkir.swimmingrecordersimple;

import android.content.Intent;
import android.os.Bundle;
import android.os.SystemClock;
import android.support.wearable.activity.WearableActivity;
import android.view.GestureDetector;
import android.view.MotionEvent;
import android.view.View;
import android.widget.TextView;

public class StartActivity extends WearableActivity {

    private TextView mTvStarting;
    private TextView mTvStartTop;
    private TextView mTvStartBottom;
    private GestureDetector mSwipe;
    private static final int SWIPE_MIN_VERTICAL_DISTANCE = 100;
    private static final int SWIPE_MAX_TIME_DIFFERENCE = 3000; // 1 second

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_start);

        // Enables Always-on
        setAmbientEnabled();

        mTvStarting = (TextView) findViewById(R.id.tvStarting);
        mTvStartTop = (TextView) findViewById(R.id.tvStartTop);
        mTvStartBottom = (TextView) findViewById(R.id.tvStartBottom);
        mTvStartTop.setText(String.format("%s", getIntent().getStringExtra("styleTv")));

        mSwipe = new GestureDetector(this, new StartSwipeDetector());
    }

    private class StartSwipeDetector extends GestureDetector.SimpleOnGestureListener {
        private boolean isLoading = false;
        private long lastSwipe = 0;

        @Override
        public boolean onScroll(MotionEvent e1, MotionEvent e2, float distanceX, float distanceY) {

            double dy = Math.sqrt(Math.pow(e2.getY()-e1.getY(), 2));

            if (dy >= SWIPE_MIN_VERTICAL_DISTANCE) {
                long dt = e1.getEventTime() - lastSwipe;
                if (dt <= SWIPE_MAX_TIME_DIFFERENCE && dt > 0 && !isLoading) {
                    mTvStarting.setVisibility(View.VISIBLE);
                    mTvStartTop.setVisibility(View.INVISIBLE);
                    mTvStartBottom.setVisibility(View.INVISIBLE);
                    isLoading = true;

                    final Intent parentIntent = getIntent();
                    Intent serviceIntent = new Intent(StartActivity.this,
                            DataLoggingService.class);
                    serviceIntent.putExtra("style", parentIntent.getStringExtra("style"));
                    serviceIntent.putExtra("body", parentIntent.getStringExtra("body"));
                    serviceIntent.putExtra("distance", parentIntent.getStringExtra("distance"));
                    startService(serviceIntent);

                    Intent stopIntent = new Intent(StartActivity.this, StopActivity.class);
                    startActivity(stopIntent);
                    finish();
                    return true;
                }
                lastSwipe = e1.getEventTime();
            }
            return false;
        }
    }

    @Override
    public boolean dispatchTouchEvent(MotionEvent event) {
        if (mSwipe != null) {
            if (mSwipe.onTouchEvent(event))
                return true;
        }
        return super.dispatchTouchEvent(event);
    }

    @Override
    public boolean onTouchEvent(MotionEvent event) {
        return mSwipe.onTouchEvent(event);
    }
}
