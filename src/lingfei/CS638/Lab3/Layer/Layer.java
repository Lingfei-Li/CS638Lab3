package lingfei.CS638.Lab3.Layer;


import lingfei.CS638.Lab3.CNN.CNN;
import lingfei.CS638.Lab3.Data.Record;
import lingfei.CS638.Lab3.Utils.Activation;
import lingfei.CS638.Lab3.Utils.MathUtil;
import lingfei.CS638.Lab3.Utils.MatrixOp;
import lingfei.CS638.Lab3.Utils.Size;

public abstract class Layer {

    public interface OutputLayer {
        public void setOutputLayerErrors(Record record);
        public int getPrediction();
    }

    /* Network structure */
    protected int outputMapsNum = 0;

    protected Size outputMapSize = null;

    protected int inputMapsNum = 0;

    protected Size inputMapSize = null;

    protected double n_in = 1;

    protected double[][][][] outputMaps = null;     //batch * outputMapsNum * outMapSize.x * outMapSize.y

    protected Size kernelSize = null;

    protected double[][][][] kernels = null;        //inMapSize * outMapSize * kernelSize.x * kernelSize.y

    protected double[][][][] adam_m = null;        //inMapSize * outMapSize * kernelSize.x * kernelSize.y

    protected double[][][][] adam_v = null;        //inMapSize * outMapSize * kernelSize.x * kernelSize.y

    protected double[][][][] errors = null;           //batch * outMapNum * outMapSize.x * outMapSize.y

    protected double[] bias = null;

    private static double adam_beta1 = 0.9;

    private static double adam_beta2 = 0.99;

    private static double adam_epsilon = 0.00000001;

    public static int batchSize = 16;
    public static int batchNum = 0;

    protected  double[][][] dropoutMask = null;

    protected  double dropoutRate = 0.5;
//    protected  double dropoutRate = 0.0;

    //Activation function for non-output layers
    protected Activation activationFunc = new Activation.LeakyReluActivation();


    /* Different layers implement different computation method */
    abstract public void computeOutput(Layer prevLayer, boolean isTraining);

    public void setHiddenLayerErrors(Layer nextLayer) {
        if(nextLayer instanceof FullyConnectedLayer) {
            assert(this.outputMapSize.equals(nextLayer.kernelSize));

            for(int i = 0; i < this.outputMapsNum; i ++) {
                MatrixOp.zeroize(this.errors[batchNum][i]);
                for(int j = 0; j < nextLayer.getOutputMapsNum(); j ++) {

                    this.errors[batchNum][i] = MatrixOp.add( this.errors[batchNum][i],
                            MatrixOp.multiplyScalar(
                                    nextLayer.kernels[i][j],
                                    nextLayer.errors[batchNum][j][0][0]
                            ));
                }
//                this.errors[batchNum][i] = MatrixOp.multiply(activationFunc.activationDeriv(this.outputMaps[batchNum][i]), this.errors[batchNum][i]);
                this.errors[batchNum][i] = MatrixOp.multiply(
                            MatrixOp.multiply(activationFunc.activationDeriv(this.outputMaps[batchNum][i]), this.errors[batchNum][i]),
                            dropoutMask[i]
                        );
            }
        }
        else if(nextLayer instanceof ConvolutionLayer) {
            for(int i = 0; i < this.outputMapsNum; i ++) {
                MatrixOp.zeroize(this.errors[batchNum][i]);
                for(int j = 0; j < nextLayer.outputMapsNum; j ++) {

                    double[][] rotatedKernel = MatrixOp.rot180(nextLayer.getKernel(i, j)) ;
                    double[][] nextError = nextLayer.errors[batchNum][j];

                    double[][] convFullResult = MatrixOp.convFull(nextError, rotatedKernel, 2);
                    this.errors[batchNum][i]  = MatrixOp.add(this.errors[batchNum][i], convFullResult);
                }
                this.errors[batchNum][i] = MatrixOp.multiply(activationFunc.activationDeriv(this.outputMaps[batchNum][i]), this.errors[batchNum][i]);
            }
        }
        else if(nextLayer instanceof  MaxPoolingLayer) {
            for(int i = 0; i < this.outputMapsNum; i ++) {
                for (int x = 0; x < nextLayer.outputMapSize.x; x ++) {
                    for (int y = 0; y < nextLayer.outputMapSize.y; y ++) {
                        for(int m = 0; m < 2; m ++) {
                            for (int n = 0; n < 2; n++) {
                                int row = x*2 + m;
                                int col = y*2 + n;
                                if(row < this.errors[batchNum][i].length && col < this.errors[batchNum][i][0].length) {
                                    this.errors[batchNum][i][row][col] = nextLayer.kernels[i][0][row][col] * nextLayer.errors[batchNum][i][x][y];
                                }
                            }
                        }
                    }
                }
                this.errors[batchNum][i] = MatrixOp.multiply(activationFunc.activationDeriv(this.outputMaps[batchNum][i]), this.errors[batchNum][i]);
            }
        }
        else {
            throw new RuntimeException("setHiddenLayerErrors not implemented for " + nextLayer.getClass().getSimpleName());
        }
    }

    public void updateBias() {
        for (int i = 0; i < this.outputMapsNum; i++) {
            double biasError = 0.0;
            for(int m = 0; m < this.outputMapSize.x; m ++) {
                for(int n = 0; n < this.outputMapSize.y; n ++) {
                    for(int b = 0; b < batchSize; b ++) {
                        biasError += this.errors[b][i][m][n];
                    }
                    biasError /= (double)batchSize;
                }
            }
            this.bias[i] += CNN.learningRate * (-1) * biasError;
        }
    }

    public void updateKernel(Layer nextLayer) {
        if(nextLayer instanceof FullyConnectedLayer) {

            assert(this.outputMapSize.equals(nextLayer.getKernelSize()));

            for(int j = 0; j < nextLayer.outputMapsNum; j ++) {
                for (int i = 0; i < this.outputMapsNum; i++) {
                    double[][] deltaKernel = new double[nextLayer.kernelSize.x][nextLayer.kernelSize.y];
                    for(int b = 0; b < batchSize; b ++) {
                        for(int m = 0; m < this.outputMapSize.x; m ++) {
                            for(int n = 0; n < this.outputMapSize.y; n ++) {
                                deltaKernel[m][n] += this.outputMaps[b][i][m][n] * nextLayer.errors[b][j][0][0];
                            }
                        }
                    }
                    deltaKernel = MatrixOp.divideScalar(deltaKernel, batchSize);

                    nextLayer.kernels[i][j] = MatrixOp.add(nextLayer.kernels[i][j], adamOptimize(nextLayer, i, j, deltaKernel));
                }
            }
        }
        else if(nextLayer instanceof ConvolutionLayer) {

            for(int j = 0; j < nextLayer.outputMapsNum; j ++) {
                for (int i = 0; i < this.outputMapsNum; i++) {
                    double[][] deltaKernel = MatrixOp.convValid(this.outputMaps[0][i], nextLayer.errors[0][j]);
                    for(int b = 1; b < batchSize; b ++) {
                        deltaKernel = MatrixOp.add(deltaKernel, MatrixOp.convValid(this.outputMaps[b][i], nextLayer.errors[b][j]));
                    }
                    deltaKernel = MatrixOp.divideScalar(deltaKernel, batchSize);

                    nextLayer.kernels[i][j] = MatrixOp.add(nextLayer.kernels[i][j], adamOptimize(nextLayer, i, j, deltaKernel));
                }
            }
        }
        else if(nextLayer instanceof  MaxPoolingLayer) {
            //No need for kernel update for Max Pooling Layer
        }
        else {
            throw new RuntimeException("setHiddenLayerErrors not implemented for " + nextLayer.getClass().getSimpleName());
        }
    }

    private double[][] adamOptimize(Layer nextLayer, int i, int j, double[][] deltaKernel) {
        nextLayer.adam_m[i][j] = MatrixOp.multiplyScalar(nextLayer.adam_m[i][j], adam_beta1);
        double[][] decayedDeltaKernel = MatrixOp.multiplyScalar(deltaKernel, 1-adam_beta1);
        nextLayer.adam_m[i][j] = MatrixOp.add(nextLayer.adam_m[i][j], decayedDeltaKernel);

        nextLayer.adam_v[i][j] = MatrixOp.multiplyScalar(nextLayer.adam_v[i][j], adam_beta2);
        double[][] decayedDeltaKernelSquared = MatrixOp.multiplyScalar(MatrixOp.multiply(deltaKernel, deltaKernel), (1-adam_beta2));
        nextLayer.adam_v[i][j] = MatrixOp.add(nextLayer.adam_v[i][j], decayedDeltaKernelSquared);

        double[][] biasCorrectedAdam_m = MatrixOp.multiplyScalar(nextLayer.adam_m[i][j], 1/(1-adam_beta1));
        double[][] biasCorrectedAdam_v = MatrixOp.multiplyScalar(nextLayer.adam_v[i][j], 1/(1-adam_beta2));

        double[][] divider = MatrixOp.addScalar(MatrixOp.sqrt(biasCorrectedAdam_v), adam_epsilon);
        biasCorrectedAdam_m = MatrixOp.divide(biasCorrectedAdam_m, divider);
        return  MatrixOp.multiplyScalar(biasCorrectedAdam_m, CNN.learningRate);
    }


    public void printKernels() {
        for(int i = 0; i < kernels.length; i ++) {
            for(int j = 0; j < kernels[0].length; j ++) {
                for(int m = 0; m < kernelSize.x; m ++) {
                    for(int n = 0; n < kernelSize.y; n ++) {
                        System.out.print(kernels[i][j][m][n] + "\t");
                    }
                }
                System.out.println();
            }
        }
    }

    /*Initialization*/
    public void initOutputMaps() { this.outputMaps = new double[batchSize][outputMapsNum][outputMapSize.x][outputMapSize.y]; }

    public void initBias() { assert(this.outputMaps != null); this.bias = MatrixOp.randomArray(outputMapsNum, n_in); }

    public void initErrors() { assert(this.outputMaps != null); this.errors = new double[batchSize][outputMapsNum][outputMapSize.x][outputMapSize.y]; }

    public void initKernels(int inputMapsNum) {
        this.kernels = new double[inputMapsNum][outputMapsNum][][];
        this.adam_m = new double[inputMapsNum][outputMapsNum][][];
        this.adam_v = new double[inputMapsNum][outputMapsNum][][];
        for(int i = 0; i < inputMapsNum; i ++) {
            for(int j = 0; j < outputMapsNum; j ++) {
                this.kernels[i][j] = activationFunc.getRandomMatrix(kernelSize.x, kernelSize.y, n_in);
                this.adam_m[i][j] = new double[kernelSize.x][kernelSize.y];
                this.adam_v[i][j] = new double[kernelSize.x][kernelSize.y];
            }
        }
    }


    /* Getter methods */
    public int getOutputMapsNum() { return outputMapsNum; }

    public Size getOutputMapSize() { return outputMapSize; }

    public Size getKernelSize() { return kernelSize; }

    public double[][] getKernel(int inputMapNum, int outputMapNum) { return kernels[inputMapNum][outputMapNum]; }


    /* Setter methods */
    public void setOutputMapsNum(int outputMapsNum) { this.outputMapsNum = outputMapsNum; }

    public void setOutputMapSize(Size outputMapSize) { this.outputMapSize = outputMapSize; }

    public void setInputmapNumAndSize(int inputMapsNum, Size inputMapSize) {
        this.inputMapsNum = inputMapsNum;
        this.inputMapSize = inputMapSize;
        this.n_in = inputMapsNum * inputMapSize.x * inputMapSize.y;
    }

    public void setKernelSize(Size kernelSize) { this.kernelSize = kernelSize; }

    protected void setOutputMap(int outputMapNum, double[][] outputMap) { this.outputMaps[batchNum][outputMapNum] = outputMap; }

    public void setAllOutputMaps(double[][][] outputMaps) { this.outputMapsNum = outputMaps.length; this.outputMaps[batchNum] = outputMaps.clone(); }

    public void setKernel(int inputMapNum, int outputMapNum, double[][] newKernel) { this.kernels[inputMapNum][outputMapNum] = newKernel; }

    public void setAllErrors(double[][][] errors) { this.errors[batchNum] = errors; }


    public void printOutputMaps() {
        for(int i = 0; i < outputMapsNum; i ++) {
            for(int m = 0; m < outputMapSize.x; m ++) {
                System.out.println();
                for(int n = 0; n < outputMapSize.y; n ++) {
                    System.out.print(this.outputMaps[i][m][n] + " ");
                }
            }
            System.out.println();
        }
    }


    // fully connected layer only
    public void resetDropoutMask() {
        if(dropoutMask == null)  {
            dropoutMask = new double[outputMapsNum][outputMapSize.x][outputMapSize.y];
            for(int j = 0; j < outputMapsNum; j ++) {
                for(int k = 0; k < outputMapSize.x; k ++) {
                    for(int m = 0; m < outputMapSize.y; m ++) {
                        dropoutMask[j][k][m] = 1;
                    }
                }
            }
        }
    }


}


