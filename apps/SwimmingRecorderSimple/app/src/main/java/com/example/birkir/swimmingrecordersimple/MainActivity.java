package com.example.birkir.swimmingrecordersimple;

import android.content.Intent;
import android.os.Bundle;
import android.support.wearable.activity.WearableActivity;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

public class MainActivity extends WearableActivity {

    private Button mBtnRecord;
    private Button mBtnSettings;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Enables Always-on
        setAmbientEnabled();

        mBtnRecord = (Button) findViewById(R.id.btnRecord);
        mBtnSettings = (Button) findViewById(R.id.btnSettings);
        mBtnSettings.setEnabled(false);

        mBtnRecord.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent recordIntent = new Intent(MainActivity.this,
                        StyleActivity.class);
                startActivity(recordIntent);
            }
        });
    }
}
