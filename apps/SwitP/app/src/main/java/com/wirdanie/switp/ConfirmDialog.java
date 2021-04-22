package com.wirdanie.switp;

import android.annotation.SuppressLint;

import androidx.appcompat.app.ActionBar;
import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;
import android.os.Handler;
import android.support.wearable.view.CircledImageView;
import android.view.MotionEvent;
import android.view.View;
import android.widget.TextView;


public class ConfirmDialog extends AppCompatActivity {
    private static final String TAG = "ConfirmActivity";

    private CircledImageView mOK;
    private CircledImageView mCancel;
    private TextView mTextView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        setContentView(R.layout.activity_confirm_dialog);

        mTextView = (TextView) findViewById(R.id.tv_question);
        mTextView.setText(getIntent().getStringExtra("Dialog_Text"));

        mOK = findViewById(R.id.btn_ok);
        mOK.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent();
                setResult(RESULT_OK, intent);
                finish();
            }
        });

        mCancel = findViewById(R.id.btn_cancel);
        mCancel.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent();
                setResult(RESULT_CANCELED, intent);
                finish();
            }
        });
    }
}
