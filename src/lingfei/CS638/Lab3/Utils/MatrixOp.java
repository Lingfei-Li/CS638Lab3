package lingfei.CS638.Lab3.Utils;

import java.io.Serializable;
import java.util.Random;
import java.util.concurrent.Callable;

public class MatrixOp {

    private static Random random = new Random();

    private interface ElementOperator {
        double operate(double value);
    }
    private static final ElementOperator sigmoid = (value) -> MathUtil.sigmoid(value);
    private static final ElementOperator sigmoidDeriv = (value) -> MathUtil.sigmoidDeriv(value);
    private static final ElementOperator relu = (value) -> MathUtil.relu(value);
    private static final ElementOperator zeroize = (value) -> 0;

    private interface BinaryElementOperator { double operateBinary(double value1, double value2); }
    private static final BinaryElementOperator add = (value1, value2) -> (value1 + value2);
    private static final BinaryElementOperator multiply = (value1, value2) -> (value1 + value2);

    /**
     * Perform the given scalar operation to each element
     * */
    private static double[][] scalarElementWiseOp(double[][] mat, double scalar, BinaryElementOperator scalarElementOp) {
        int w = mat.length;
        int h = mat[0].length;
        double[][] result = new double[w][h];
        for(int i = 0; i < w; i ++) {
            for(int j = 0; j < h; j ++) {
                result[i][j] = scalarElementOp.operateBinary(mat[i][j], scalar);
            }
        }
        return result;
    }

    /**
     * Perform the given operation to each element
     * */
    private static double[][] elementWiseOp(double[][] mat, ElementOperator elementOp) {
        int w = mat.length;
        int h = mat[0].length;
        double[][] result = new double[w][h];
        for(int i = 0; i < w; i ++) {
            for(int j = 0; j < h; j ++) {
                result[i][j] = elementOp.operate(mat[i][j]);
            }
        }
        return result;
    }


    /**
     * Perform the given binary operation on the two matrices element-wise
     * */
    private static double[][] binaryElementWiseOp(double[][] m1, double[][]m2, BinaryElementOperator binElemOp) {
        if(m1.length != m2.length || m1[0].length != m2[0].length) {
            throw new IllegalArgumentException("Dimensions of the input matrices don't match");
        }
        int w = m1.length;
        int h = m1[0].length;
        double[][] result = new double[w][h];
        for(int i = 0; i < w; i ++) {
            for(int j = 0; j < h; j ++) {
                result[i][j] = binElemOp.operateBinary(m1[i][j], m2[i][j]);
            }
        }
        return result;
    }

    /**
     * Add the two give matrices element-wise
     * */
    public static double[][] add(double[][] m1, double[][]m2) { return binaryElementWiseOp(m1, m2, add); }

    /**
     * Multiply the two give matrices element-wise
     * */
    public static double[][] multiply(double[][] m1, double[][]m2) { return binaryElementWiseOp(m1, m2, multiply); }

    /**
     * Rotate the given matrix by 180 degrees
     * */
    public static double[][] rot180(double[][] mat) {
        int w = mat.length;
        int h = mat[0].length;
        double[][] result = new double[w][h];
        for(int i = 0; i < w; i ++) {
            for(int j = 0; j < h; j ++) {
                result[w-i-1][h-j-1] = mat[i][j];
            }
        }
        return result;
    }

    /**
     * Add a number to all matrix elements
     * */
    public static double[][] addScalar(double[][] mat, double scalar) { return scalarElementWiseOp(mat, scalar, add); }

    /**
     * Multiply a number to all matrix elements
     * */
    public static double[][] multiplyScalar(double[][] mat, double scalar) { return scalarElementWiseOp(mat, scalar, multiply); }

    /**
     * Apply sigmoid to all matrix elements
     * */
    public static double[][] sigmoid(double[][] mat) { return elementWiseOp(mat, sigmoid); }

    /**
     * Apply sigmoid derivative to all matrix elements
     * */
    public static double[][] sigmoidDeriv(double[][] mat) { return elementWiseOp(mat, sigmoidDeriv); }

    /**
     * Set all matrix elements to zero
     * */
    public static double[][] zeroize(double[][] mat) { return elementWiseOp(mat, zeroize); }


        /**
         * Apply rectified linear operation to all matrix elements
         * */
    public static double[][] relu(double[][] mat) { return elementWiseOp(mat, relu); }

    /**
     * Sum all input matrix and return the result
     * */
    public static double sum(double[][] mat) {
        int w = mat.length;
        int h = mat[0].length;
        double result = 0;
        for(int i = 0; i < w; i ++) {
            for(int j = 0; j < h; j ++) {
                result = mat[i][j];
            }
        }
        return result;
    }

    /**
     * Convolution operation without zero padding
     * */
    public static double[][] convValid(double[][] input, double[][] kernel) {
        //Input size
        int m = input.length;
        int n = input[0].length;

        //Kernel size
        int km = kernel.length;
        int kn = kernel[0].length;

        //Feature map size
        int fm = m - km + 1;
        int fn = n - kn + 1;

        double[][] featureMap = new double[fm][fn];

        //Stride = 1
        for(int i = 0; i < fm; i ++) {
            for(int j = 0; j < fn; j ++) {
                double sum = 0.0;
                for(int ki = 0; ki < km; ki ++) {
                    for(int kj = 0; kj < kn; kj ++) {
                        sum += input[i+ki][j+kj] * kernel[ki][kj];
                    }
                }
                featureMap[i][j] = sum;

            }
        }
        return featureMap;
    }

    /**
     * Convolution operation with given zero padding
     * */
    public static double[][] convFull(double[][] input, double[][] kernel, int pm, int pn) {
        //Stride
        int s = 1;

        //Input size
        int m = input.length;
        int n = input[0].length;

        //Kernel size
        int km = kernel.length;
        int kn = kernel[0].length;

        //Feature map size
        int fm = (m - km + 2*pm)/s + 1;
        int fn = (n - kn + 2*pn)/s + 1;

        double[][] featureMap = new double[fm][fn];

        //i, j: feature map index
        for(int i = 0; i < fm; i ++) {
            for(int j = 0; j < fn; j ++) {
                double sum = 0.0;
                for(int ki = 0; ki < km; ki ++) {
                    for(int kj = 0; kj < kn; kj ++) {
                        Size inputIdx = new Size(i - pm + ki, j - pn + kj);
                        if(inputIdx.x < 0 || inputIdx.x >= m || inputIdx.y < 0 || inputIdx.y >= n) {
                            sum += 0;
                        }
                        else {
                            sum += input[inputIdx.x][inputIdx.y] * kernel[ki][kj];
                        }
                    }
                }
                featureMap[i][j] = sum;
            }
        }
        return featureMap;
    }

    /**
     * Convolution operation with customized zero padding. Same padding for all edges
     * */
    public static double[][] convFull(double[][] input, double[][] kernel, int p) {
        return convFull(input, kernel, p, p);
    }

    /**
     * Convolution operation with zero padding. output in the same size as input
     * */
    public static double[][] convFull(double[][] input, double[][] kernel) {
        //zero padding
        int pm = (kernel.length - 1)/2;
        int pn = (kernel[0].length - 1)/2;
        return convFull(input, kernel, pm, pn);
    }

    public static double[] randomArray(int len) {
        double[] array = new double[len];
        for (int i = 0; i < len; i++) {
            array[i] = MathUtil.randomWeight();
        }
        return array;
    }

    public static double[][] randomMatrix(int x, int y) {
        double[][] matrix = new double[x][y];
        for (int i = 0; i < x; i++) {
            for (int j = 0; j < y; j++) {
                matrix[i][j] = MathUtil.randomWeight();
            }
        }
        return matrix;
    }

    public static void printMat(double[][] mat) {
        int w = mat.length;
        int h = mat[0].length;
        for(int i = 0; i < w; i ++) {
            for(int j = 0; j < h; j ++) {
                System.out.print(mat[i][j] + "\t");
            }
            System.out.println();
        }
        System.out.println();
    }

    public static void printDim(String name, double[][] mat) {
        System.out.println(name + " dimension: " + mat.length + " * " + mat[0].length);
    }

    public static void printDim(String name, double[][][] mat) {
        System.out.println(name + " dimension: " + mat.length + " * " + mat[0].length + " * " + mat[0][0].length);
    }

    public static void printDim(String name, double[][][][] mat) {
        System.out.println(name + " dimension: " + mat.length + " * " + mat[0].length + " * " + mat[0][0].length + " * " + mat[0][0][0].length);
    }

    //Testing
    public static void main(String[] args) {
        System.out.println("Running MathUtil tests");
        int w = 10, h = 10;
        double[][] mat = new double[w][h];
        for(int i = 0; i < w; i ++) {
            for(int j = 0; j < h; j ++) {
                mat[i][j] = 1;
            }
        }

        double[][] kernel = new double[3][3];
        for(int i = 0; i < 3; i ++) {
            for(int j = 0; j < 3; j ++) {
                kernel[i][j] = 1;
            }
        }

        printMat(mat);
        printMat(kernel);
        printMat(convFull(mat, kernel));


    }

}
