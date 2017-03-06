package lingfei.CS638.Lab3.Layer;

import lingfei.CS638.Lab3.Utils.Activation;
import lingfei.CS638.Lab3.Utils.MatrixOp;
import lingfei.CS638.Lab3.Utils.Size;

public class ConvolutionLayer extends Layer {

    public ConvolutionLayer(int outputMapsNum, Size kernelSize) {
        this.kernelSize = kernelSize;
        this.outputMapsNum = outputMapsNum;
    }

    public ConvolutionLayer(int outputMapsNum, Size kernelSize, Activation activationFunc) {
        this.kernelSize = kernelSize;
        this.outputMapsNum = outputMapsNum;
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
        for(int j = 0; j < this.outputMapsNum; j ++) {
            double[][] curOutputMap = null;
            for(int i = 0; i < prevLayer.outputMapsNum; i ++) {
//                MatrixOp.printMat(this.getKernel(i, j));
//                MatrixOp.printMat(MatrixOp.convValid( prevLayer.getOutputMap(i), getKernel(i, j)));
                if(curOutputMap == null) {
                    curOutputMap = MatrixOp.convValid( prevLayer.getOutputMap(i), this.getKernel(i, j));
                } else {
                    MatrixOp.add(curOutputMap, MatrixOp.convValid( prevLayer.getOutputMap(i), getKernel(i, j)) );
                }
            }

            this.setOutputMap(j, activationFunc.activation(curOutputMap));

//            MatrixOp.printMat(this.outputMaps[j]);
//            MatrixOp.printMat(activationFunc.activation(this.outputMaps[j]));
        }
    }
}




