package lingfei.CS638.Lab3.Layer;

import lingfei.CS638.Lab3.Utils.MatrixOp;
import lingfei.CS638.Lab3.Utils.Size;

public class ConvolutionLayer extends Layer {

    public ConvolutionLayer(int outputMapsNum, Size kernelSize) {
        this.kernelSize = kernelSize;
        this.outputMapsNum = outputMapsNum;
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
        for(int j = 0; j < this.outputMapsNum; j ++) {
            double[][] curOutputMap = null;
            for(int i = 0; i < prevLayer.outputMapsNum; i ++) {
                if(curOutputMap == null) {
                    curOutputMap = MatrixOp.convValid( prevLayer.getOutputMap(i), this.getKernel(i, j));
                } else {
                    MatrixOp.add(curOutputMap, MatrixOp.convValid( prevLayer.getOutputMap(i), getKernel(i, j)) );
                }
            }
            this.setOutputMap(j, MatrixOp.sigmoid(curOutputMap));
        }
    }
}




