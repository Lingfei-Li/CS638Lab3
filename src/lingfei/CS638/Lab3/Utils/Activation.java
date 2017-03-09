package lingfei.CS638.Lab3.Utils;

/**
 * Created by Lingfei on 2017-3-2.
 */
public interface Activation {

    double activation(double t);
    double activationDeriv(double t);
    double[][] activation(double[][] mat);
    double[][] activationDeriv(double[][] mat);
    double getRandomWeight(double n_in);
    double[] getRandomArray(int len, double n_in);
    double[][] getRandomMatrix(int x, int y, double n_in);


    public class SigmoidActivation implements Activation {
        public double activation(double t) { return MathUtil.sigmoid(t); }
        public double activationDeriv(double t) { return MathUtil.sigmoidDeriv(t); }
        public double[][] activation(double[][] mat) { return MatrixOp.sigmoid(mat); }
        public double[][] activationDeriv(double[][] mat) { return MatrixOp.sigmoidDeriv(mat); }
        public double getRandomWeight(double n_in) { return MathUtil.xavier_weight(n_in); }
        public double[] getRandomArray(int len, double n_in) { return MatrixOp.randomArray(len, n_in); }
        public double[][] getRandomMatrix(int x, int y, double n_in) { return MatrixOp.randomMatrix(x, y, n_in); }
    }

    public class ReluActivation implements Activation {
        public double activation(double t) { return MathUtil.relu(t); }
        public double activationDeriv(double t) { return MathUtil.reluDeriv(t); }
        public double[][] activation(double[][] mat) { return MatrixOp.relu(mat); }
        public double[][] activationDeriv(double[][] mat) { return MatrixOp.reluDeriv(mat); }
        public double getRandomWeight(double n_in) { return MathUtil.xavier_weight_positive(n_in); }
        public double[] getRandomArray(int len, double n_in) { return MatrixOp.positiveRandomArray(len, n_in); }
        public double[][] getRandomMatrix(int x, int y, double n_in) { return MatrixOp.positiveRandomMatrix(x, y, n_in); }
    }
    public class LeakyReluActivation implements Activation {
        public double activation(double t) { return MathUtil.reluLeaky(t); }
        public double activationDeriv(double t) { return MathUtil.reluLeakyDeriv(t); }
        public double[][] activation(double[][] mat) { return MatrixOp.reluLeaky(mat); }
        public double[][] activationDeriv(double[][] mat) { return MatrixOp.reluLeakyDeriv(mat); }
        public double getRandomWeight(double n_in) { return MathUtil.xavier_weight_positive(n_in); }
        public double[] getRandomArray(int len, double n_in) { return MatrixOp.positiveRandomArray(len, n_in); }
        public double[][] getRandomMatrix(int x, int y, double n_in) { return MatrixOp.positiveRandomMatrix(x, y, n_in); }
    }


}
