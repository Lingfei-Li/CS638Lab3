package lingfei.CS638.Lab3.Layer;

import lingfei.CS638.Lab3.CNN.CNN;
import lingfei.CS638.Lab3.Data.Record;
import lingfei.CS638.Lab3.Utils.MathUtil;
import lingfei.CS638.Lab3.Utils.MatrixOp;

public class FCOutputLayer extends FullyConnectedLayer implements Layer.OutputLayer {

    public FCOutputLayer(int outputMapsNum) { super(outputMapsNum); }

    public void setOutputLayerErrors(Record record) {

        double[][][] errors = new double[outputMapsNum][1][1];

        for(int i = 0; i < outputMapsNum; i ++) {
            double teacher = 0.0;
            if(i == record.label) {
                teacher = 1.0;
            }
            errors[i][0][0] = teacher - outputMaps[batchNum][i][0][0];
        }

        setAllErrors(errors);
    }

    public void computeOutput(Layer prevLayer, boolean isTraining) {
        //Each input element should match with one kernel element
        assert(prevLayer.outputMapSize.equals(this.kernelSize));

        for(int j = 0; j < this.outputMapsNum; j ++) {
            double[][] sumMat = new double[1][1];
            sumMat[0][0] = 0.0;
            for(int i = 0; i < prevLayer.outputMapsNum; i ++) {
                if(isTraining) {
                    double change = MatrixOp.sum(MatrixOp.multiply(prevLayer.outputMaps[batchNum][i], this.getKernel(i, j)));
                    sumMat[0][0] += change;
                }
                else {
                    double change = MatrixOp.sum(MatrixOp.multiply(prevLayer.outputMaps[batchNum][i],
                            MatrixOp.multiplyScalar(this.getKernel(i, j), (1-dropoutRate))
                            ));
                    sumMat[0][0] += change;
                }
            }

            sumMat[0][0] += (-1) * this.bias[j];

            this.setOutputMap(j, MatrixOp.sigmoid(sumMat));

        }
    }

    public int getPrediction() {
        double[] output = new double[outputMapsNum];
        for(int i = 0; i < outputMapsNum; i ++) {
            output[i] = outputMaps[batchNum][i][0][0];
//            System.out.print(output[i] + " ");
        }
//        System.out.println();
        return MathUtil.argmax(output);
    }



}

