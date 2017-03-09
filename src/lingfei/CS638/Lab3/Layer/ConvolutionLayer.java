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

    public void computeOutput(Layer prevLayer, boolean isTraining) {
        for(int j = 0; j < this.outputMapsNum; j ++) {
            double[][] curOutputMap = null;
            for(int i = 0; i < prevLayer.outputMapsNum; i ++) {
                if(curOutputMap == null) {
                    if(isTraining) {
                        curOutputMap = MatrixOp.convValid( prevLayer.outputMaps[batchNum][i], this.getKernel(i, j));
                    }
                    else {
                        curOutputMap = MatrixOp.convValid( prevLayer.outputMaps[batchNum][i],
                                MatrixOp.multiplyScalar( this.getKernel(i, j), (1-dropoutRate) )
                        );
                    }
                } else {
                    if(isTraining) {
                        curOutputMap = MatrixOp.add(curOutputMap, MatrixOp.convValid( prevLayer.outputMaps[batchNum][i], getKernel(i, j)) );
                    }
                    else {
                        curOutputMap = MatrixOp.add(curOutputMap, MatrixOp.convValid( prevLayer.outputMaps[batchNum][i],
                                MatrixOp.multiplyScalar( this.getKernel(i, j), (1-dropoutRate))
                        ));
                    }
                }
            }

            this.setOutputMap(j, activationFunc.activation(curOutputMap));
        }
    }
}




