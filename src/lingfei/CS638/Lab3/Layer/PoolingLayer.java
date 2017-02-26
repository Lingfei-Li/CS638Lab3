package lingfei.CS638.Lab3.Layer;

public class PoolingLayer extends Layer {
    public void computeOutput(Layer prevLayer) {
        throw new IllegalArgumentException("computeOutput not implemented for Pooling Layer yet");
    }

    public void initKernels(int inputMapsNum) {
        System.out.println("Note: initKernels is not implemented for Pooling layer yet");
    }
}
