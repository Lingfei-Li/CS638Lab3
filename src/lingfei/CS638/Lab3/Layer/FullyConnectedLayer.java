package lingfei.CS638.Lab3.Layer;


import lingfei.CS638.Lab3.CNN.CNN;
import lingfei.CS638.Lab3.Utils.Activation;
import lingfei.CS638.Lab3.Utils.MathUtil;
import lingfei.CS638.Lab3.Utils.MatrixOp;
import lingfei.CS638.Lab3.Utils.Size;

public class FullyConnectedLayer extends Layer{

    public FullyConnectedLayer(int outputMapsNum) {
        this.outputMapsNum = outputMapsNum;
        this.outputMapSize = new Size(1, 1);
    }

    public void computeOutput(Layer prevLayer, boolean isTraining) {
        //Each input element should match with one kernel element
        assert(prevLayer.outputMapSize.equals(this.kernelSize));

        for(int j = 0; j < this.outputMapsNum; j ++) {
            double[][] sumMat = new double[1][1];
            sumMat[0][0] = 0.0;
            for(int i = 0; i < prevLayer.outputMapsNum; i ++) {
                sumMat[0][0] += MatrixOp.sum(MatrixOp.multiply(prevLayer.outputMaps[batchNum][i], this.kernels[i][j]));
            }

            sumMat[0][0] += (-1) * this.bias[j];

            if(isTraining) {
                this.setOutputMap(j,
                        MatrixOp.multiply(
                                activationFunc.activation(sumMat),
                                dropoutMask[j]
                        )
                );
            }
            else {
                this.setOutputMap(j,
                        MatrixOp.multiplyScalar(
                                activationFunc.activation(sumMat),
                                (1-dropoutRate)
                        )
                );
            }
        }
    }

    public void resetDropoutMask() {
        dropoutMask = new double[outputMapsNum][outputMapSize.x][outputMapSize.y];
        for(int j = 0; j < outputMapsNum; j ++) {
            for(int k = 0; k < outputMapSize.x; k ++) {
                for(int m = 0; m < outputMapSize.y; m ++) {
                    if(MathUtil.flipCoin(dropoutRate)) {
                        dropoutMask[j][k][m] = 0;
                    }
                    else {
                        dropoutMask[j][k][m] = 1;
                    }
                }
            }
        }
    }


}
