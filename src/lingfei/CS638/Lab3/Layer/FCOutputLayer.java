package lingfei.CS638.Lab3.Layer;

import lingfei.CS638.Lab3.CNN.CNN;
import lingfei.CS638.Lab3.Data.Record;
import lingfei.CS638.Lab3.Utils.MathUtil;

public class FCOutputLayer extends FullyConnectedLayer implements Layer.OutputLayer {

    public FCOutputLayer(int outputMapsNum) { super(outputMapsNum); }

    public void setOutputLayerErrors(Record record) {

        double[][][] errors = new double[outputMapsNum][1][1];

        for(int i = 0; i < outputMapsNum; i ++) {
            double teacher = 0.0;
            if(i == record.label) {
                teacher = 1.0;
            }
            errors[i][0][0] = CNN.activationFunc(getOutputMap(i)[0][0]) * (teacher - getOutputMap(i)[0][0]);
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

