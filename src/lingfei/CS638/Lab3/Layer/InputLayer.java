package lingfei.CS638.Lab3.Layer;

import lingfei.CS638.Lab3.Utils.Size;

public class InputLayer extends Layer {

    public InputLayer(int outputMapsNum, Size outputMapSize) {
        this.outputMapsNum = outputMapsNum;
        this.outputMapSize = outputMapSize;
    }

    public void computeOutput(Layer prevLayer) {
        throw new IllegalArgumentException("computeOutput method of the input layer is unexpectedly called");
    }

    public void initKernels(int inputMapsNum) {
        throw new IllegalArgumentException("initKernels method of the input layer is unexpectedly called");
    }

}
