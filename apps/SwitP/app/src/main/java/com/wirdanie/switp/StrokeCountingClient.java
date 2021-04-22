package com.wirdanie.switp;

import android.util.Log;

import java.util.function.Consumer;

class StrokeCountingClient {
    private static final String TAG = "StrokeCountingClient";

    private int lag;
    private Float dynthreshold;
    private Float statthreshold;
    private Float influence;
    private Boolean lastSignal;

    private Float[] filteredDelayLine;
    private int count = 0;

    private boolean init = false;

    Consumer<Long> retFun;

    StrokeCountingClient(int lag, float dynthreshold, float statthreshold, float influence, Consumer<Long> retFun) {
        this.lag = lag;
        this.dynthreshold = dynthreshold;
        this.statthreshold = statthreshold;
        this.influence = influence;
        this.filteredDelayLine = new Float[this.lag];

        this.lastSignal = false;
        this.retFun = retFun;
    }

    // Uses the z channel, kept the array sample structure for compatibility
    // Initialise arrays
    void addSample(Long timestamp, Float[] newSampleArray){
        Float newSample = newSampleArray[2]; //z-channel
        Log.d(TAG, "newVal: " + newSample);
        // If the delayLine has not been initialised yet
        if(!init) {
            this.filteredDelayLine[this.count] = new Float(newSample);
            this.count = (++this.count) % this.filteredDelayLine.length;
            if(this.count == 0){
                this.init = true;
            }

        } else {
            Boolean actSignal = false;

            float sum = 0;
            float ss = 0;

            for (float val : this.filteredDelayLine) {
                sum += val;
                ss += val * val;
            }

            float avg = (float) sum / ((float) lag);
            float std_dev = (float) Math.sqrt(ss / ((float) lag) - avg * avg);



            // Signal out of current bounds -> Peak
            if (Math.abs(newSample - avg) > Math.max(this.dynthreshold * std_dev, this.statthreshold)){
                actSignal = true;
                // Reduce influence of signal / filter it
                int prevIndex = ((this.count - 1) + this.filteredDelayLine.length) % this.filteredDelayLine.length;
                this.filteredDelayLine[this.count] = newSample * this.influence - this.filteredDelayLine[prevIndex] * (1 - influence);
            } else {
                // Signal not peaking / don't filter it
                this.filteredDelayLine[this.count] = newSample;
            }

            // Update index for next iteration
            this.count = (++this.count) % this.filteredDelayLine.length;

            // Rising edge on the detector output (beginning of positive or negative peak)
            if (actSignal && !this.lastSignal) {
                this.returnValues(timestamp);
            }
            this.lastSignal = actSignal;
        }
    }

    private void returnValues(Long timestamp) {
        this.retFun.accept(timestamp);
    }
}
