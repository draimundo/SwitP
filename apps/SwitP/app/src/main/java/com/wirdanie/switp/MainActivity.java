/*
Copyright 2016 The Android Open Source Project

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
 */
package com.wirdanie.switp;

import android.annotation.SuppressLint;
import android.content.Intent;
import android.os.Bundle;

import androidx.preference.SwitchPreference;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;
import android.widget.ImageView;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;
import androidx.preference.PreferenceFragmentCompat;
import androidx.wear.ambient.AmbientModeSupport;
import android.support.wearable.view.CircledImageView;
import androidx.wear.widget.CircularProgressLayout;
import androidx.wear.widget.drawer.WearableActionDrawerView;
import androidx.wear.widget.drawer.WearableDrawerLayout;

public class MainActivity extends AppCompatActivity implements
        AmbientModeSupport.AmbientCallbackProvider{

    private static final String TAG = "MainActivity";

    private WearableActionDrawerView mWearableActionDrawer;
    private WearableDrawerLayout mWearableDrawerLayout;
    private ScrollView mScrollView;

    private CircularProgressLayout mStartProgress;
    private CircledImageView mExit;

    private ImageView mStartStop;
    private TextView mLabel;

    @SuppressLint("ClickableViewAccessibility")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        Log.d(TAG, "onCreate()");

        setContentView(R.layout.activity_main);

        // Enables Ambient mode.
        AmbientModeSupport.attach(this);

        mScrollView = findViewById(R.id.scroll_view_settings);

        // Bottom Action Drawer
        mWearableActionDrawer = findViewById(R.id.bottom_action_drawer);
        // Peeks action drawer on the bottom.
        mWearableActionDrawer.getController().peekDrawer();

        mWearableDrawerLayout = findViewById(R.id.drawer_layout);
        getSupportFragmentManager()
                .beginTransaction()
                .replace(R.id.drawer_content, new SettingsPrefsFragment())
                .commit();
        mWearableDrawerLayout.setDrawerStateCallback(new WearableDrawerLayout.DrawerStateCallback() {
            @Override
            public void onDrawerStateChanged(WearableDrawerLayout layout, int newState) {
                super.onDrawerStateChanged(layout, newState);

                if(newState == 0 && !mWearableActionDrawer.isOpened()){
                    mScrollView.scrollTo(0,0);
                }
            }
        });

        mStartStop = findViewById(R.id.start_iv);
        mLabel = findViewById(R.id.start_tv);

        mStartProgress = findViewById(R.id.start_progress);
        mStartProgress.setOnTimerFinishedListener(new CircularProgressLayout.OnTimerFinishedListener() {
            @Override
            public void onTimerFinished(CircularProgressLayout layout) {
                mLabel.setText("Recording...");
                mStartStop.setImageResource(R.drawable.ic_stop_24px);
                mStartProgress.setBackgroundColor(getResources().getColor(android.R.color.holo_red_dark));

                mWearableActionDrawer.getController().closeDrawer();
                mWearableActionDrawer.setVisibility(View.GONE);

                mStartProgress.stopTimer();

                Intent recordIntent = new Intent(MainActivity.this, RecordingActivity.class);
                startActivity(recordIntent);

                Toast toast = Toast.makeText(getApplicationContext(),"Started Recording", Toast.LENGTH_SHORT);
                toast.show();
            }
        });
        mStartProgress.setOnTouchListener(new View.OnTouchListener() {
            @Override
            public boolean onTouch(View view, MotionEvent motionEvent) {
                int action = motionEvent.getAction();
                Log.d(TAG, "Action:" + action);
                switch(action){
                    case MotionEvent.ACTION_DOWN:
                    case MotionEvent.ACTION_MOVE:
                        mStartProgress.setTotalTime(2000);
                        mStartProgress.startTimer();
                    break;
                    default:
                        mStartProgress.stopTimer();
                }
                return true;
            }
        });

        mExit = findViewById(R.id.btn_exit);
        mExit.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                createDialog("Exit SwitP?");
            }
        });
    }

    @Override
    protected void onActivityResult (int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        // Collect data from the intent and use it
        if(resultCode == RESULT_OK && requestCode == 1){
            Toast toast = Toast.makeText(getApplicationContext(),"Goodbye!", Toast.LENGTH_SHORT);
            toast.show();
            finishAndRemoveTask();
        }
    }

    private void createDialog(String DialogQuestion){
        Intent intent = new Intent(this, ConfirmDialog.class);
        intent.putExtra("Dialog_Text", DialogQuestion);
        int requestCode = 1; // Or some number you choose
        startActivityForResult(intent, requestCode);
    }

    @Override
    public AmbientModeSupport.AmbientCallback getAmbientCallback() {
        return new MyAmbientCallback();
    }

    @Override
    protected void onResume() {
        mLabel.setText("Record");
        mStartStop.setImageResource(R.drawable.ic_play_arrow_24px);
        mStartProgress.setBackgroundColor(getResources().getColor(R.color.swimming_blue));
        mWearableActionDrawer.setVisibility(View.VISIBLE);
        mWearableActionDrawer.getController().peekDrawer();
        super.onResume();
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

    public static class SettingsPrefsFragment extends PreferenceFragmentCompat {
        private static final String TAG = "NotificationsActivity";

        @Override
        public void onCreate(Bundle savedInstanceState) {
            super.onCreate(savedInstanceState);

        }

        @Override
        public void onCreatePreferences(Bundle savedInstanceState, String rootKey) {
            addPreferencesFromResource(R.xml.prefs_settings);
        }

        public void onResume() {
            super.onResume();


            getPreferenceManager().setSharedPreferencesName(getString(R.string.preference_file_key));
            getPreferenceManager().setSharedPreferencesMode(MODE_PRIVATE);

            final SwitchPreference mSimulateSwitchPref =
                    (SwitchPreference) findPreference(getString(R.string.key_pref_simulate));
            final SwitchPreference mSaveOriginalSwitchPref =
                    (SwitchPreference) findPreference(getString(R.string.key_pref_save_original));
            final SwitchPreference mSaveResampledSwitchPref =
                    (SwitchPreference) findPreference(getString(R.string.key_pref_save_resampled));
            final SwitchPreference mSaveClassifiedSwitchPref =
                    (SwitchPreference) findPreference(getString(R.string.key_pref_save_classified));
            final SwitchPreference mSaveLapSwitchPref =
                    (SwitchPreference) findPreference(getString(R.string.key_pref_save_lap));
            final SwitchPreference mSaveStrokeSwitchPref =
                    (SwitchPreference) findPreference(getString(R.string.key_pref_save_stroke));
            final SwitchPreference mLengthSwitchPref =
                    (SwitchPreference) findPreference(getString(R.string.key_pref_pool_length));


            initSave(mSimulateSwitchPref);
            initSave(mSaveOriginalSwitchPref);
            initSave(mSaveResampledSwitchPref);
            initSave(mSaveClassifiedSwitchPref);
            initSave(mSaveLapSwitchPref);
            initSave(mSaveStrokeSwitchPref);
            initSave(mLengthSwitchPref);
        }

        void initSave(SwitchPreference switchPref) {
            switchPref.setChecked(this.getPreferenceManager().getSharedPreferences().getBoolean(switchPref.getKey(),true));
        }
    }
}