package lingfei.CS638.Lab3.Layer;


import lingfei.CS638.Lab3.CNN.CNN;
import lingfei.CS638.Lab3.Data.Record;
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

    protected double[][][] outputMaps = null;     //outputMapsNum * outMapSize.x * outMapSize.y

    protected Size kernelSize = null;

    protected double[][][][] kernels = null;        //inMapSize * outMapSize * kernelSize.x * kernelSize.y

    protected double[][][] errors = null;           //outMapNum * outMapSize.x * outMapSize.y

    protected double[] bias = null;



    /* Different layers implement different computation method */
    abstract public void computeOutput(Layer prevLayer);

    public void setHiddenLayerErrors(Layer nextLayer) {
        if(nextLayer instanceof FullyConnectedLayer) {

            assert(this.outputMapSize.equals(nextLayer.kernelSize));

            for(int i = 0; i < this.outputMapsNum; i ++) {
                MatrixOp.zeroize(this.errors[i]);
                for(int j = 0; j < nextLayer.getOutputMapsNum(); j ++) {
                    for(int m = 0; m < nextLayer.getKernelSize().x; m ++) {
                        for(int n = 0; n < nextLayer.getKernelSize().y; n ++) {
                            this.errors[i][m][n] += nextLayer.kernels[i][j][m][n] * nextLayer.errors[j][0][0];
                        }
                    }
                }
                this.errors[i] = MatrixOp.multiply(MatrixOp.sigmoidDeriv(this.outputMaps[i]), this.errors[i]);
            }
        }
        else if(nextLayer instanceof ConvolutionLayer) {
            for(int i = 0; i < this.outputMapsNum; i ++) {
                MatrixOp.zeroize(this.errors[i]);
                for(int j = 0; j < nextLayer.outputMapsNum; j ++) {

                    double[][] rotatedKernel = MatrixOp.rot180(nextLayer.getKernel(i, j)) ;
                    double[][] nextError = nextLayer.errors[j];

                    double[][] convFullResult = MatrixOp.convFull(nextError, rotatedKernel, 2);
                    this.errors[i]  = MatrixOp.add(this.errors[i], convFullResult);
                }
                this.errors[i] = MatrixOp.multiply(MatrixOp.sigmoidDeriv(this.outputMaps[i]), this.errors[i]);
            }
        }
        else if(nextLayer instanceof  PoolingLayer) {
            System.out.println("Next layer is pooling layer");
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
                    biasError += this.getErrors()[i][m][n];
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
                    for(int m = 0; m < this.outputMapSize.x; m ++) {
                        for(int n = 0; n < this.outputMapSize.y; n ++) {
//                            nextLayer.kernels[i][j][m][n] += CNN.learningRate * this.getOutputMap(i)[m][n] * nextLayer.getErrors()[j][0][0];
                            deltaKernel[m][n] += CNN.learningRate * this.outputMaps[i][m][n] * nextLayer.errors[j][0][0];
                        }
                    }
                    nextLayer.kernels[i][j] = MatrixOp.add(nextLayer.kernels[i][j], deltaKernel);
                }
            }
        }
        else if(nextLayer instanceof ConvolutionLayer) {

            for(int j = 0; j < nextLayer.outputMapsNum; j ++) {
                for (int i = 0; i < this.outputMapsNum; i++) {
                    double[][] deltaKernel = MatrixOp.convValid(this.getOutputMap(i), nextLayer.getErrors()[j]);

                    deltaKernel = MatrixOp.multiplyScalar(deltaKernel, CNN.learningRate);

                    nextLayer.setKernel(i, j, MatrixOp.add(nextLayer.getKernel(i, j), deltaKernel));
                }
            }
        }
        else if(nextLayer instanceof  PoolingLayer) {
            System.out.println("Next layer is pooling layer");
        }
        else {
            throw new RuntimeException("setHiddenLayerErrors not implemented for " + nextLayer.getClass().getSimpleName());
        }
    }


    public void printKernels() {
        for(int i = 0; i < kernels.length; i ++) {
            for(int j = 0; j < kernels[0].length; j ++) {
                for(int m = 0; m < kernelSize.x; m ++) {
                    for(int n = 0; n < kernelSize.y; n ++) {
                        System.out.println(kernels[i][j][m][n] + "\t");
                    }
                }
                System.out.println();
            }
        }
    }

    /*Initialization*/
    public void initOutputMaps() { this.outputMaps = new double[outputMapsNum][outputMapSize.x][outputMapSize.y]; }

    public void initBias() { assert(this.outputMaps != null); this.bias = MatrixOp.randomArray(outputMapsNum); }

    public void initErrors() { assert(this.outputMaps != null); this.errors = new double[outputMapsNum][outputMapSize.x][outputMapSize.y]; }

    abstract public void initKernels(int inputMapsNum);


    /* Getter methods */
    public int getOutputMapsNum() { return outputMapsNum; }

    public Size getOutputMapSize() { return outputMapSize; }

    public Size getKernelSize() { return kernelSize; }

    public double[][] getOutputMap(int mapNum) { return outputMaps[mapNum]; }

    public double[][] getKernel(int inputMapNum, int outputMapNum) { return kernels[inputMapNum][outputMapNum]; }

    public double[][][] getErrors() { return errors; }


    /* Setter methods */
    public void setOutputMapsNum(int outputMapsNum) { this.outputMapsNum = outputMapsNum; }

    public void setOutputMapSize(Size outputMapSize) { this.outputMapSize = outputMapSize; }

    public void setKernelSize(Size kernelSize) { this.kernelSize = kernelSize; }

    protected void setOutputMap(int outputMapNum, double[][] outputMap) { this.outputMaps[outputMapNum] = outputMap; }

    public void setAllOutputMaps(double[][][] outputMaps) { this.outputMapsNum = outputMaps.length; this.outputMaps = outputMaps.clone(); }

    public void setKernel(int inputMapNum, int outputMapNum, double[][] newKernel) { this.kernels[inputMapNum][outputMapNum] = newKernel; }

    public void setAllErrors(double[][][] errors) { this.errors = errors; }


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


}


