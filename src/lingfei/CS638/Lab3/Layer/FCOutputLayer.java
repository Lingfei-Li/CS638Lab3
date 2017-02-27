package lingfei.CS638.Lab3.Layer;

import lingfei.CS638.Lab3.Data.Record;
import lingfei.CS638.Lab3.Utils.MathUtil;

public class FCOutputLayer extends FullyConnectedLayer implements Layer.OutputLayer {

    public FCOutputLayer(int outputMapsNum) { super(outputMapsNum); }

    public void setOutputLayerErrors(Record record) {
        double[] oneHotArray = MathUtil.buildOneHotArray(outputMapsNum, record.label);

        double[][][] errors = new double[outputMapsNum][1][1];

        for(int i = 0; i < outputMapsNum; i ++) {
            errors[i][0][0] = MathUtil.sigmoidDeriv(getOutputMap(i)[0][0]) * (oneHotArray[i] - getOutputMap(i)[0][0]);
//            errors[i][0][0] = MathUtil.reluDeriv(getOutputMap(i)[0][0]) * (oneHotArray[i] - getOutputMap(i)[0][0]);
        }

        setAllErrors(errors);
    }

    public int getPrediction() {
        double[] output = new double[outputMapsNum];
        for(int i = 0; i < outputMapsNum; i ++) {
            output[i] = getOutputMap(i)[0][0];
//            System.out.print(output[i] + " ");
        }
//        System.out.println();
        return MathUtil.argmax(output);
    }



}

