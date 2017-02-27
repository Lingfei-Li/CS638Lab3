package lingfei.CS638.Lab3.Layer;


import lingfei.CS638.Lab3.Utils.MatrixOp;
import lingfei.CS638.Lab3.Utils.Size;

public class FullyConnectedLayer extends Layer{


    public FullyConnectedLayer(int outputMapsNum) {
        this.outputMapsNum = outputMapsNum;
        this.outputMapSize = new Size(1, 1);
    }


    public void initKernels(int inputMapsNum) {
        this.kernels = new double[inputMapsNum][outputMapsNum][][];
        for(int i = 0; i < inputMapsNum; i ++) {
            for(int j = 0; j < outputMapsNum; j ++) {
                this.kernels[i][j] = MatrixOp.randomMatrix(kernelSize.x, kernelSize.y);
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

            this.setOutputMap(j, MatrixOp.sigmoid(sumMat));
//            this.setOutputMap(j, MatrixOp.relu(sumMat));
        }
    }


}
