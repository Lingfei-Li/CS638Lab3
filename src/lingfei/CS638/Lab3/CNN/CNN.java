package lingfei.CS638.Lab3.CNN;

import lingfei.CS638.Lab3.Data.*;
import lingfei.CS638.Lab3.Layer.*;
import lingfei.CS638.Lab3.Utils.*;

import java.util.ArrayList;
import java.util.List;


public class CNN {

    private int inputWidth, inputHeight, inputDepth;

    private List<Record> trainset, tuneset, testset;
    private List<Layer> layers;
    private FCOutputLayer outputLayer;

    public final static double learningRate = 0.01;


    public CNN(Dataset trainset, Dataset tuneset, Dataset testset) {
        this.inputWidth = trainset.getImages().get(0).getWidth();
        this.inputHeight = trainset.getImages().get(0).getHeight();
        this.inputDepth = 4;

        this.trainset = new ArrayList<>();
        for(Instance instance : trainset.getImages()) {
            double[][][] data = instance.getAllChannelMatrix();
            double label = instance.getLabelAsDouble();
            Record record = new Record(data, label);
            this.trainset.add(record);
        }

        this.tuneset = new ArrayList<>();
        for(Instance instance : tuneset.getImages()) {
            double[][][] data = instance.getAllChannelMatrix();
            double label = instance.getLabelAsDouble();
            Record record = new Record(data, label);
            this.tuneset.add(record);
        }

        this.testset = new ArrayList<>();
        for(Instance instance : testset.getImages()) {
            double[][][] data = instance.getAllChannelMatrix();
            double label = instance.getLabelAsDouble();
            Record record = new Record(data, label);
            this.testset.add(record);
        }


        //Things to be initialized:
        // outputMapsNum, outputMapSize, outputMap
        // kernelSize

        this.layers = new ArrayList<>();
        this.layers.add(new InputLayer(inputDepth, new Size(inputHeight, inputWidth)));             //params: (# of channels, size of each image)
//        this.layers.add(new ConvolutionLayer(4, new Size(3, 3)));             //params: (# of filters, size of each filter)
        this.layers.add(new FullyConnectedLayer(10));                            //# of hidden units
        this.layers.add(outputLayer = new FCOutputLayer(6));                    //# of outputs (categories)

        setup();

    }


    public void train() {
        int maxEpoch = 1000;
        for(int epoch = 0; epoch < maxEpoch; epoch ++) {
//            int batchSize = 50;
//            int[] perm = MathUtil.genPerm(this.trainset.size(), batchSize);
            int classNum = 6;
            int[][] confusionMatrix = new int[classNum][classNum];
            double acc = 0.0;
            for (int i = 0; i < this.trainset.size(); i++) {
//                Record record = this.trainset.get(perm[i]);
                Record record = this.trainset.get(i);
                forward(record);
                backprop(record);
                updateParams();

                int prediction = outputLayer.getPrediction();
                confusionMatrix[record.label][prediction] ++;
                if(prediction == record.label) {
                    acc ++;
                }
            }
            acc /= this.trainset.size();
            System.out.println("Accuracy: " + acc);
            for (int i = 0; i < confusionMatrix.length; i++) {
                for (int j = 0; j < confusionMatrix[0].length; j++) {
                    System.out.print("\t" + confusionMatrix[i][j]);
                }
                System.out.println();
            }
//            test(this.trainset);
        }
    }

    public void test(List<Record> ds) {
        int classNum = 6;
        int[][] confusionMatrix = new int[classNum][classNum];
        double acc = 0.0;
        for(int i = 0; i < ds.size(); i ++) {
            Record record = ds.get(i);
            forward(record);
            int prediction = outputLayer.getPrediction();
            confusionMatrix[record.label][prediction] ++;
            if(prediction == record.label) {
                acc ++;
            }
        }
        acc /= ds.size();
        System.out.println("Accuracy: " + acc);
        for (int i = 0; i < confusionMatrix.length; i++) {
            for (int j = 0; j < confusionMatrix[0].length; j++) {
                System.out.print("\t" + confusionMatrix[i][j]);
            }
            System.out.println();
        }
    }

    public void forward(Record record) {
        this.layers.get(0).setAllOutputMaps(record.data);

        for(int l = 1; l < layers.size(); l ++) {
            Layer prevLayer = layers.get(l-1);
            Layer curLayer = layers.get(l);
            curLayer.computeOutput(prevLayer);
        }
    }

    public void backprop(Record record) {
        //set error
        outputLayer.setOutputLayerErrors(record);
        for(int l = layers.size()-2; l >= 0; l --) {
            layers.get(l).setHiddenLayerErrors(layers.get(l+1));
        }
    }

    public void updateParams() {
        //update parameters
        for(int l = layers.size()-2; l >= 0; l --) {
            layers.get(l).updateKernel(layers.get(l+1));
        }

        for(int l = layers.size()-1; l >= 0; l --) {
            layers.get(l).updateBias();
        }
    }

    /**
     * Conducts initialization of outputMapSize, outputMaps, kernels
     * */
    private void setup() {
        layers.get(0).setOutputMapsNum(4);

        for(int i = 0; i < layers.size(); i ++) {
            Layer curLayer = layers.get(i);
            if(i != 0) {
                Layer prevLayer = layers.get(i-1);

                //Init kernels
                int inputMapsNum = prevLayer.getOutputMapsNum();
                if(curLayer instanceof ConvolutionLayer) {
                    curLayer.initKernels(inputMapsNum);
                    Size curOutputMapSize = prevLayer.getOutputMapSize().minus(curLayer.getKernelSize()).plus(1);
                    curLayer.setOutputMapSize(curOutputMapSize);
                }
                else if(curLayer instanceof FullyConnectedLayer){
                    curLayer.setKernelSize(prevLayer.getOutputMapSize());
                    curLayer.initKernels(inputMapsNum);
//                curLayer.setOutputMapSize(new Size(1, 1));
                }
                else {
                    throw new RuntimeException("ERROR: setup not implemented for " + curLayer.getClass().getSimpleName());
                }
            }
            //Init outputMap and error matrix. BatchSize must be set at first
            curLayer.initOutputMaps();
            curLayer.initBias();
            curLayer.initErrors();
        }
    }

}
