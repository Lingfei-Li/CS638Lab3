package lingfei.CS638.Lab3.Layer;

import lingfei.CS638.Lab3.Data.Record;
import lingfei.CS638.Lab3.Utils.MathUtil;

public class FCOutputLayer extends FullyConnectedLayer implements Layer.OutputLayer {

    public FCOutputLayer(int outputMapsNum) { super(outputMapsNum); }

    public void setOutputLayerErrors(Record record) {
        double[] oneHotArray = MathUtil.buildOneHotArray(outputMapsNum, record.label);

        double[][][] errors = new double[outputMapsNum][1][1];

        for(int i = 0; i < outputMapsNum; i ++) {
            errors[i][0][0] = MathUtil.sigmoidDeriv(outputMaps[i][0][0]) * (oneHotArray[i] - outputMaps[i][0][0]);
        }

        setAllErrors(errors);
    }

    public int getPrediction() {
        double[] output = new double[outputMapsNum];
        for(int i = 0; i < outputMapsNum; i ++) {
            output[i] = outputMaps[i][0][0];
        }
        return MathUtil.argmax(output);
    }



}

