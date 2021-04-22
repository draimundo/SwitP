package com.wirdanie.switp;

import android.os.AsyncTask;
import android.util.Log;

import java.util.concurrent.ExecutorService;
import java.util.function.BiConsumer;

import static java.util.concurrent.Executors.newSingleThreadExecutor;

public class FilteringClient {
    private String TAG = "FilteringClient";

    class FIRFilter {
        ExecutorService singleThreadExecutor;
        Float[] impulseResponse;
        Float[][] delayLine;
        int nChannels;
        int count = 0;
        BiConsumer<Long, Float[]> retFun = null;

        FIRFilter(Float[] coefs, int nChannels, BiConsumer<Long, Float[]> retFun) {
            this.impulseResponse = coefs.clone();
            this.nChannels = nChannels;
            this.delayLine = new Float[this.nChannels][this.impulseResponse.length * 2]; //longer to avoid overwriting during filtering
            this.singleThreadExecutor = newSingleThreadExecutor(); //single executor for this specific filter

            for (int i = 0; i < this.delayLine.length; i++) {
                for (int j = 0; j < this.delayLine[i].length; j++) {
                    this.delayLine[i][j] = new Float(0f);
                }
            }

            this.retFun = retFun;
        }

        void addSample(Float[] newSample) {
            for (int i_channel = 0; i_channel < this.nChannels; i_channel++) {
                this.delayLine[i_channel][this.count] = newSample[i_channel];
            }
            this.count = (++this.count) % this.delayLine[0].length;
        }

        void naiveCompute(Long timestamp) {
            // Perform convolution on copy of delayLine
            NaiveComputeTask naiveComp = new NaiveComputeTask(this, timestamp, this.count);
            naiveComp.executeOnExecutor(this.singleThreadExecutor);
        }

        void returnValues(Long timestamp, Float[] values) {
            retFun.accept(timestamp, values);
        }
    }

    private class NaiveComputeTask extends AsyncTask<Void, Void, Float[]> {
        private FIRFilter filter = null;
        private Long timestamp = null;
        private int count = 0;

        public NaiveComputeTask(FIRFilter filter, Long timestamp, int count) {
            this.filter = filter;
            this.timestamp = timestamp;
            this.count = count;
        }

        protected Float[] doInBackground(Void... none) {
            // Perform convolution
            Float ret[] = new Float[this.filter.nChannels];
            for (int i_channel = 0; i_channel < this.filter.delayLine.length; i_channel++) {
                ret[i_channel] = new Float(0f);
                int i_sample = this.count;
                for (int i_resp = 0; i_resp < this.filter.impulseResponse.length; i_resp++) {
                    ret[i_channel] += this.filter.delayLine[i_channel][i_sample--] * this.filter.impulseResponse[i_resp];
                    if (i_sample < 0) {
                        i_sample = this.filter.delayLine[i_channel].length - 1;
                    }
                }
            }
            return ret;
        }

        protected void onPostExecute(Float[] result) {
            this.filter.returnValues(this.timestamp, result);
        }
    }

    public class Resampler {
        private int upFactor;
        private int downFactor;
        private FIRFilter filter;
        private int nChannels;

        // Note: values returned are upFactor smaller
        Resampler(int upFactor, int downFactor, int origFreq, int nChannels, BiConsumer<Long, Float[]> retFun) {
            this.upFactor = upFactor;
            this.downFactor = downFactor;
            this.nChannels = nChannels;
            this.filter = new FIRFilter(FIRdefinitions.getCoeffs(origFreq), nChannels, retFun);
        }


        public void resample(long timestamp, Float[] newSample) {
            this.addSample(timestamp, newSample); // add sample to filter delayline
            for (int i = 0; i < this.upFactor - 1; i++) { //perform upsampling
                Float[] tmp = new Float[this.nChannels];
                for (int i_channel = 0; i_channel < this.nChannels; i_channel++) {
                    tmp[i_channel] = new Float(0f);
                }
                this.addSample(timestamp, tmp);
            }
        }

        private int k = 0; // variable for downsampling

        private void addSample(long timestamp, Float[] newSample) {
            this.filter.addSample(newSample);

            this.k = (++this.k) % this.downFactor; //compute only when necessary (not dropped samples)
            if (k == 0) {
                this.filter.naiveCompute(timestamp); // timestamp is the same for results of oversampling
            }
        }
    }

    static class FIRdefinitions {
        //gives a filter with cutoff around 15Hz, for a given frequency, and group delay ~195ms
        static Float[] getCoeffs(int origFreq) {
            switch (origFreq) {
                case 50:
                    return FIRdefinitions.lowpass_50Hz;
                case 52:
                    return FIRdefinitions.lowpass_52Hz;
                case 100:
                    return FIRdefinitions.lowpass_100Hz;
                case 104:
                    return FIRdefinitions.lowpass_104Hz;
                case 300:
                    return FIRdefinitions.lowpass_300Hz;
                default:
                    return null;
            }
        }


        private final static Float[] lowpass_50Hz = new Float[]{
            0.03300579265f, -0.01842423715f, -0.01444907766f, -0.01036530454f,-0.005231866613f,
            0.0008306825184f, 0.006587534212f,    0.010178186f, 0.009979614057f, 0.005496318452f,
            -0.002123276005f, -0.01022768486f, -0.01549922395f, -0.01523117349f,-0.008569524623f,
            0.002805912402f,  0.01493254211f,  0.02279038355f,  0.02215995826f,  0.01148436684f,
            -0.006984101143f, -0.02727553435f, -0.04117656127f, -0.04071139544f, -0.02088731527f,
            0.01822920702f,  0.07086315006f,   0.1266429126f,   0.1732317954f,   0.1997164786f,
            0.1997164786f,   0.1732317954f,   0.1266429126f,  0.07086315006f,  0.01822920702f,
            -0.02088731527f, -0.04071139544f, -0.04117656127f, -0.02727553435f,-0.006984101143f,
            0.01148436684f,  0.02215995826f,  0.02279038355f,  0.01493254211f, 0.002805912402f,
            -0.008569524623f, -0.01523117349f, -0.01549922395f, -0.01022768486f,-0.002123276005f,
            0.005496318452f, 0.009979614057f,    0.010178186f, 0.006587534212f,0.0008306825184f,
            -0.005231866613f, -0.01036530454f, -0.01444907766f, -0.01842423715f,  0.03300579265f
        };

        private final static Float[] lowpass_52Hz = new Float[]{
            0.0330257453f, -0.02001899853f, -0.01505003031f, -0.01037893258f,-0.005125209223f,
            0.000693711394f, 0.006124267355f, 0.009657185525f, 0.009904264472f, 0.006320724729f,
            -0.0003285394923f,-0.007983822376f, -0.01384980232f,  -0.0153529821f, -0.01117823273f,
            -0.001989313867f, 0.009492048994f,  0.01917269826f,  0.02290240489f,  0.01802146435f,
            0.004629361443f,  -0.0139226187f, -0.03157154098f, -0.04107448086f, -0.03613418341f,
            -0.01349371485f,  0.02566460148f,  0.07551784813f,   0.1267437786f,   0.1687522978f,
            0.1923978925f,   0.1923978925f,   0.1687522978f,   0.1267437786f,  0.07551784813f,
            0.02566460148f, -0.01349371485f, -0.03613418341f, -0.04107448086f, -0.03157154098f,
            -0.0139226187f, 0.004629361443f,  0.01802146435f,  0.02290240489f,  0.01917269826f,
            0.009492048994f,-0.001989313867f, -0.01117823273f,  -0.0153529821f, -0.01384980232f,
            -0.007983822376f,-0.0003285394923f, 0.006320724729f, 0.009904264472f, 0.009657185525f,
            0.006124267355f, 0.000693711394f,-0.005125209223f, -0.01037893258f, -0.01505003031f,
            -0.02001899853f,   0.0330257453f
        };

        private final static Float[] lowpass_100Hz = new Float[]{
            0.03812275827f, -0.01215263177f, -0.01044638921f,-0.008974706754f,-0.007630094886f,
            -0.006315203849f,-0.004955168813f,-0.003507825313f,-0.001968629193f,-0.0003716844949f,
            0.001212454983f, 0.002685589716f, 0.003933733795f, 0.004840542097f, 0.005303423386f,
            0.005249022972f, 0.004645187873f, 0.003510082141f,  0.00191684009f,-8.850536688e-06f,
            -0.002097034594f,-0.004146286752f,-0.005941809621f,-0.007277213503f,-0.007976293564f,
            -0.007913276553f,-0.007030427922f,-0.005349966232f,-0.002978117671f,-0.0001012940411f,
            0.003025730839f,  0.00609974144f, 0.008795627393f,  0.01079712342f,  0.01182903256f,
            0.01168799773f,   0.0102690421f, 0.007585217245f,  0.00377756753f,-0.0008862063987f,
            -0.006024421193f, -0.01116501354f, -0.01577916928f, -0.01932234317f, -0.02127902023f,
            -0.02120778896f, -0.01878318936f, -0.01383035444f,-0.006349038333f, 0.003475377336f,
            0.01527531724f,  0.02851902321f,  0.04254206643f,  0.05659035593f,  0.06987152994f,
            0.08161081374f,  0.09110671282f,  0.09778206795f,   0.1012263969f,   0.1012263969f,
            0.09778206795f,  0.09110671282f,  0.08161081374f,  0.06987152994f,  0.05659035593f,
            0.04254206643f,  0.02851902321f,  0.01527531724f, 0.003475377336f,-0.006349038333f,
            -0.01383035444f, -0.01878318936f, -0.02120778896f, -0.02127902023f, -0.01932234317f,
            -0.01577916928f, -0.01116501354f,-0.006024421193f,-0.0008862063987f,  0.00377756753f,
            0.007585217245f,   0.0102690421f,  0.01168799773f,  0.01182903256f,  0.01079712342f,
            0.008795627393f,  0.00609974144f, 0.003025730839f,-0.0001012940411f,-0.002978117671f,
            -0.005349966232f,-0.007030427922f,-0.007913276553f,-0.007976293564f,-0.007277213503f,
            -0.005941809621f,-0.004146286752f,-0.002097034594f,-8.850536688e-06f,  0.00191684009f,
            0.003510082141f, 0.004645187873f, 0.005249022972f, 0.005303423386f, 0.004840542097f,
            0.003933733795f, 0.002685589716f, 0.001212454983f,-0.0003716844949f,-0.001968629193f,
            -0.003507825313f,-0.004955168813f,-0.006315203849f,-0.007630094886f,-0.008974706754f,
            -0.01044638921f, -0.01215263177f,  0.03812275827f
        };

        private final static Float[] lowpass_104Hz = new Float[]{
            0.03552071005f, -0.01708847843f, -0.01357261837f, -0.01054840628f, -0.00759743806f,
            -0.004468453117f,-0.001137749758f, 0.002178874798f, 0.005080319941f, 0.007077627815f,
            0.007730066776f, 0.006776094437f, 0.004239084199f,0.0004745071055f,-0.003857203992f,
            -0.007890458219f, -0.01071582921f, -0.01157311816f, -0.01003318559f,-0.006133040413f,
            -0.000425423088f, 0.006076053716f,  0.01205327921f,  0.01612396911f,  0.01712463237f,
            0.0143809393f, 0.007913973182f,-0.001466081594f, -0.01220853906f, -0.02221530676f,
            -0.02915603854f, -0.03085747361f, -0.02570087276f, -0.01295684557f, 0.007005456369f,
            0.03267864883f,  0.06156483293f,   0.0904944241f,   0.1160673648f,   0.1351487488f,
            0.145339027f,    0.145339027f,   0.1351487488f,   0.1160673648f,   0.0904944241f,
            0.06156483293f,  0.03267864883f, 0.007005456369f, -0.01295684557f, -0.02570087276f,
            -0.03085747361f, -0.02915603854f, -0.02221530676f, -0.01220853906f,-0.001466081594f,
            0.007913973182f,   0.0143809393f,  0.01712463237f,  0.01612396911f,  0.01205327921f,
            0.006076053716f,-0.000425423088f,-0.006133040413f, -0.01003318559f, -0.01157311816f,
            -0.01071582921f,-0.007890458219f,-0.003857203992f,0.0004745071055f, 0.004239084199f,
            0.006776094437f, 0.007730066776f, 0.007077627815f, 0.005080319941f, 0.002178874798f,
            -0.001137749758f,-0.004468453117f, -0.00759743806f, -0.01054840628f, -0.01357261837f,
            -0.01708847843f,  0.03552071005f
        };

        // This one was designed for the StrokeCountingClient
        // -> Not necessarily the same group delay
        private final static Float[] lowpass_300Hz = new Float[]{
            0.01482923701f, 0.003652780317f, 0.003656572429f,  0.00331811863f, 0.002606982831f,
            0.001516116899f,6.587486132e-05f,-0.001693449449f,-0.003680974245f,-0.005786493886f,
            -0.007874095812f, -0.00978731364f, -0.01135715283f, -0.01241017785f, -0.01277951244f,
            -0.01231417526f, -0.01089043263f,-0.008420759812f,-0.004861887079f,-0.0002211896062f,
            0.005440477747f,  0.01200778224f,  0.01931338757f,  0.02714382671f,  0.03524788469f,
            0.04334690049f,  0.05114815012f,  0.05835854635f,  0.06469900161f,   0.0699185282f,
            0.07380694896f,  0.07620583475f,  0.07701659203f,  0.07620583475f,  0.07380694896f,
            0.0699185282f,  0.06469900161f,  0.05835854635f,  0.05114815012f,  0.04334690049f,
            0.03524788469f,  0.02714382671f,  0.01931338757f,  0.01200778224f, 0.005440477747f,
            -0.0002211896062f,-0.004861887079f,-0.008420759812f, -0.01089043263f, -0.01231417526f,
            -0.01277951244f, -0.01241017785f, -0.01135715283f, -0.00978731364f,-0.007874095812f,
            -0.005786493886f,-0.003680974245f,-0.001693449449f,6.587486132e-05f, 0.001516116899f,
            0.002606982831f,  0.00331811863f, 0.003656572429f, 0.003652780317f,  0.01482923701f
        };

    }
}
