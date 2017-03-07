package lingfei.CS638.Lab3.Utils;

import java.util.HashSet;
import java.util.Random;
import java.util.Set;

public class MathUtil {

    private static Random rand = new Random();


    public static boolean flipCoin(double probability) { return rand.nextDouble() < probability; }

    public static double sigmoid(double t) { return 1.0/(1.0+Math.exp(-t)); }

    public static double sigmoidDeriv(double sigmoidVal) { return sigmoidVal*(1-sigmoidVal); }

    public static double relu(double t) { return Math.max(t, 0); }

    public static double reluDeriv(double val) { return (val > 0) ? 1 : 0; }

    public static double reluLeaky(double t) {
        if(t < 0) {
            return 0.1*t;
        }
        return t;
    }
    public static double reluLeakyDeriv(double val) { return (val > 0) ? 1 : 0.1; }

    /**
     * Produce a random number in the range of [-0.05, 0.05), for weight initialization
     * */
    public static double randomWeight() { return (rand.nextDouble() - 0.5)/100; }
    public static double positiveRandomWeight() { return rand.nextDouble()/100; }

    /**
     * Get the index of the maximum element in the array
     * */
    public static int argmax(double[] array) {
        double maxVal = array[0];
        int maxValIndex = 0;
        for(int i = 1; i < array.length; i ++) {
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

    /**
     * Generate a randomly permuted array of given size within the given range
     * */
    public static int[] genPerm(int bound, int size) {
        Set<Integer> set = new HashSet<>();
        while (set.size() < size) {
            set.add(rand.nextInt(bound));
        }
        int[] randPerm = new int[size];
        int i = 0;
        for (Integer value : set) {
            randPerm[i++] = value;
        }
        return randPerm;
    }

}

