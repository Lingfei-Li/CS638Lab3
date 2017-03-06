package lingfei.CS638.Lab3.Layer;


import lingfei.CS638.Lab3.CNN.CNN;
import lingfei.CS638.Lab3.Utils.Activation;
import lingfei.CS638.Lab3.Utils.MatrixOp;
import lingfei.CS638.Lab3.Utils.Size;

public class FullyConnectedLayer extends Layer{


    public FullyConnectedLayer(int outputMapsNum) {
        this.outputMapsNum = outputMapsNum;
        this.outputMapSize = new Size(1, 1);
    }

    public FullyConnectedLayer(int outputMapsNum, Activation activationFunc) {
        this.outputMapsNum = outputMapsNum;
        this.outputMapSize = new Size(1, 1);
        this.activationFunc = activationFunc;
    }


    public void initKernels(int inputMapsNum) {
        this.kernels = new double[inputMapsNum][outputMapsNum][][];
        for(int i = 0; i < inputMapsNum; i ++) {
            for(int j = 0; j < outputMapsNum; j ++) {
                this.kernels[i][j] = activationFunc.getRandomMatrix(kernelSize.x, kernelSize.y);
            }
        }
    }


    public void computeOutput(Layer prevLayer) {
        //Each input element should match with one kernel element
        assert(prevLayer.outputMapSize.equals(this.kernelSize));

        for(int j = 0; j < this.outputMapsNum; j ++) {
            double[][] sumMat = new double[1][1];
            sumMat[0][0] = 0.0;
            for(int i = 0; i < prevLayer.outputMapsNum; i ++) {
                double change = MatrixOp.sum(MatrixOp.multiply(prevLayer.getOutputMap(i), this.getKernel(i, j)));
                sumMat[0][0] += change;
            }


            sumMat[0][0] += (-1) * this.bias[j];

            this.setOutputMap(j, activationFunc.activation(sumMat));
        }
    }


}
