package lingfei.CS638.Lab3.Utils;

import java.util.Random;

public class MathUtil {

    private static Random rand = new Random();


    public static boolean flipCoin(double probability) { return rand.nextDouble() < probability; }

    public static double sigmoid(double t) { return 1.0/(1.0+Math.exp(-t)); }

    public static double sigmoidDeriv(double sigmoidVal) { return sigmoidVal*(1-sigmoidVal); }

    public static double relu(double t) { return Math.max(t, 0); }

    public static double reluDeriv(double val) { return (val > 0) ? 1 : 0; }

    /**
     * Produce a random number in the range of [-0.05, 0.05), for weight initialization
     * */
    public static double randomWeight() { return (rand.nextDouble() - 0.5); }

    /**
     * Get the index of the maximum element in the array
     * */
    public static int argmax(double[] array) {
        double maxVal = Double.MIN_VALUE;
        int maxValIndex = -1;
        for(int i = 0; i < array.length; i ++) {
            if(array[i] > maxVal) {
                maxVal = array[i];
                maxValIndex = i;
            }
        }
        return maxValIndex;
    }

    /**
     * Build and return an one-hot array
     * */
    public static double[] buildOneHotArray(int len, int hotPos) {
        double[] array = new double[len];
        array[hotPos] = 1;
        return array;
    }
}

