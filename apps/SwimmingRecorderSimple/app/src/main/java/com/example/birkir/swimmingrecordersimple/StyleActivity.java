package com.example.birkir.swimmingrecordersimple;

import android.content.Intent;
import android.os.Bundle;
import android.support.wearable.activity.WearableActivity;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

public class StyleActivity extends WearableActivity {

    private Button mBtnFreestyle;
    private Button mBtnBreaststroke;
    private Button mBtnBackstroke;
    private Button mBtnButterfly;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_style);

        // Enables Always-on
        setAmbientEnabled();

        mBtnFreestyle = (Button) findViewById(R.id.btnFreestyle);
        mBtnBreaststroke = (Button) findViewById(R.id.btnBreaststroke);
        mBtnBackstroke = (Button) findViewById(R.id.btnBackstroke);
        mBtnButterfly = (Button) findViewById(R.id.btnButterfly);

        mBtnFreestyle.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                toStart(getString(R.string.fileFreestyle), getString(R.string.tvFreestyle));
            }
        });
        mBtnBreaststroke.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                toStart(getString(R.string.fileBreaststroke), getString(R.string.tvBreaststroke));
            }
        });
        mBtnBackstroke.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                toStart(getString(R.string.fileBackstroke), getString(R.string.tvBackstroke));
            }
        });
        mBtnButterfly.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                toStart(getString(R.string.fileButterfly), getString(R.string.tvButterfly));
            }
        });
    }

    private void toStart(String style, String styleTv) {
        Intent startIntent = new Intent(StyleActivity.this, StartActivity.class);
        startIntent.putExtra("style", style);
        startIntent.putExtra("styleTv",styleTv);
        startActivity(startIntent);
    }
}
