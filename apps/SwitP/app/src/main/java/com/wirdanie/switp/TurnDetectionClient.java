package com.wirdanie.switp;

import android.os.AsyncTask;
import android.util.Log;

import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentNavigableMap;
import java.util.concurrent.ConcurrentSkipListMap;
import java.util.concurrent.ExecutorService;
import java.util.function.Consumer;

import static java.util.concurrent.Executors.newSingleThreadExecutor;

public class TurnDetectionClient {
    private static Integer[] TransPredThres_perc = new Integer[]{50, 45, 40, 35, 30, 25};
    private static Integer[] MinTransDuration_s = new Integer[]{6, 5, 4, 3, 2, 1};

    private static final String TAG = "TurnDetectionClient";
    private static final Long NANOSECONDS_PER_SECOND = (long) 1E9;

    private static class TransitionPair implements Comparable<TransitionPair> {
        Float TransPredThres;
        Long MinTransDuration_ns;

        TransitionPair(Float TransPredThres, Long MinTransDuration_ns) {
            this.TransPredThres = TransPredThres;
            this.MinTransDuration_ns = MinTransDuration_ns;
        }

        @Override
        public int compareTo(TransitionPair o) {
            return Math.round(o.TransPredThres * o.MinTransDuration_ns - this.TransPredThres * this.MinTransDuration_ns); //other way around for descending order
        }
    }

    private static List<TransitionPair> TransitionPairList = new LinkedList<>();

    static {
        for (Integer TransPredThres_i : TransPredThres_perc) {
            for (Integer MinTransDuration_s_i : MinTransDuration_s) {
                TransitionPairList.add(new TransitionPair(((float) TransPredThres_i)/100.0f,MinTransDuration_s_i*NANOSECONDS_PER_SECOND));
            }
        }
        Collections.sort(TransitionPairList, TransitionPair::compareTo);
    }

    private static enum eState {
        WaitForCl0,
        WaitForTmin,
        TurnDetected;
        static public final Integer nStates = 1 + TurnDetected.ordinal();
    }

    public class LapDetector {
        private Long MinLapTimens;
        private Long MaxLapTimens;
        Consumer<Long> retFun = null;
        private Long lastTimestamp = 0l;
        private Long firstTimestamp = 0l;
        private eState actState;
        ExecutorService singleThreadExecutor;
        ConcurrentSkipListMap<Long, Float> probMap = new ConcurrentSkipListMap<Long, Float>();

        LapDetector(Long MinLapTimens, Long MaxLapTimens, Consumer<Long> retFun) {
            this.MinLapTimens = MinLapTimens;
            this.MaxLapTimens = MaxLapTimens;
            this.retFun = retFun;
            this.actState = eState.TurnDetected;
            this.singleThreadExecutor = newSingleThreadExecutor();
        }

        private void transition(boolean cl0, Long timestamp) {
            switch (actState) {
                case WaitForCl0:
                    if (cl0) {
                        this.actState = eState.WaitForTmin;
                        this.firstTimestamp = timestamp;
                    } else if (timestamp - this.lastTimestamp >= this.MaxLapTimens) {
                        new LapFindingTask(this.probMap.headMap(timestamp, true), this).executeOnExecutor(this.singleThreadExecutor);
                    }
                    break;
                case WaitForTmin:
                    if (cl0){
                        if (timestamp - this.firstTimestamp >= this.MinLapTimens) {
                            this.actState = eState.TurnDetected;
                            this.returnValue(timestamp);
                        }
                    } else {
                        this.actState = eState.WaitForCl0;
                    }
                    break;
                case TurnDetected:
                    if (!cl0) {
                        this.actState = eState.WaitForCl0;
                    }
                    break;
                default:
            }
        }

        void clearMap() {
            this.probMap.clear();
        }

        void addMeasurement(Long timestamp, Float probability) {
            this.probMap.put(timestamp, probability);
            this.transition((probability >= 0.5), timestamp);
        }

        void returnValue(Long timestamp) {
            Log.d(TAG, timestamp + ": Turn returned");
            this.lastTimestamp = timestamp;
            this.retFun.accept(timestamp);
            this.clearMap();
        }
    }

    private class LapFindingTask extends AsyncTask<Void, Void, Long> {
        ConcurrentNavigableMap<Long, Float> probMap;
        LapDetector lapDetector;

        LapFindingTask(ConcurrentNavigableMap<Long, Float> probMap, LapDetector lapDetector) {
            this.probMap = probMap;
            this.lapDetector = lapDetector;
        }

        @Override
        protected Long doInBackground(Void... voids) {
            Log.d(TAG, "LapFindingTask");
            Long timeLength = 0L;
            Long lastTime = 0L;
            Long ret = 0L;
            outer:
            for (TransitionPair tPair : TransitionPairList) {
                for (Map.Entry<Long, Float> entry : this.probMap.entrySet()) {
                    if (entry.getValue() >= tPair.TransPredThres){
                        if (lastTime > 0L) {
                            timeLength += entry.getKey() - lastTime;
                            if (timeLength >= tPair.MinTransDuration_ns) {
                                ret = entry.getKey();
                                break outer;
                            }
                        }
                        lastTime = entry.getKey();
                    } else {
                        lastTime = 0L;
                    }
                }
            }
            return ret;
        }

        @Override
        protected void onPostExecute(Long transitionTime) {
            if (transitionTime != 0L) {
                this.lapDetector.returnValue(transitionTime);
            }
        }
    }

    public TurnDetectionClient() {

    }

}
