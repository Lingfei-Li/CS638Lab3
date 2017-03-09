package lingfei.CS638.Lab3.Layer;

import lingfei.CS638.Lab3.Utils.MatrixOp;

public class MaxPoolingLayer extends Layer {

    public void initKernels(int inputMapsNum) {
        //In max pooling, the kernel is used as an indicator of which input serves as the max
        this.kernels = new double[inputMapsNum][1][kernelSize.x][kernelSize.y];
    }

    public void computeOutput(Layer prevLayer, boolean isTraining) {
        //Down sampling
        for(int i = 0; i < prevLayer.outputMapsNum; i ++) {
            MatrixOp.zeroize(this.kernels[i][0]);
            for(int x = 0; x < prevLayer.outputMapSize.x; x += 2) {
                for(int y = 0; y < prevLayer.outputMapSize.y; y += 2) {
                    double maxVal = prevLayer.outputMaps[batchNum][i][x][y];
                    int maxX = x, maxY = y;
                    for(int m = 0; m < 2; m ++) {
                        for(int n = 0; n < 2; n ++) {
                            int row = x + m;
                            int col = y + n;

                            if(row < prevLayer.outputMapSize.x && col < prevLayer.outputMapSize.y) {
                                double val = prevLayer.outputMaps[batchNum][i][row][col];
                                if(val > maxVal) {
                                    maxVal = val;
                                    maxX = row;
                                    maxY = col;
                                }
                            }
                        }
                    }
                    this.kernels[i][0][maxX][maxY] = 1;
                    this.outputMaps[batchNum][i][x/2][y/2] = maxVal;
                }
            }
        }

    }

}
