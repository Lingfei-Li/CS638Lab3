package lingfei.CS638.Lab3.CNN;

import lingfei.CS638.Lab3.Data.*;
import lingfei.CS638.Lab3.Layer.*;
import lingfei.CS638.Lab3.Utils.*;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;


public class CNN {

    private int inputWidth, inputHeight, inputDepth;

    private List<Record> trainset, tuneset, testset;
    private List<Layer> layers;
    private FCOutputLayer outputLayer;
    private List<Double> trainCurve;
    private List<Double> tuneCurve;

    public final static double learningRate = 0.001;


    public CNN(Dataset trainset, Dataset tuneset, Dataset testset) {
        this.inputWidth = trainset.getImages().get(0).getWidth();
        this.inputHeight = trainset.getImages().get(0).getHeight();
        this.inputDepth = 3;

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


        //Add layers to the network
        this.layers = new ArrayList<>();
        this.layers.add(new InputLayer(inputDepth,
                new Size(inputHeight, inputWidth)));                    //params: (# of channels, size of each image)


//        this.layers.add(new ConvolutionLayer(30, new Size(3, 3)));       //params: (# of filters, size of each filter)
//        this.layers.add(new MaxPoolingLayer());

//        this.layers.add(new ConvolutionLayer(30, new Size(3, 3)));       //params: (# of filters, size of each filter)
//        this.layers.add(new MaxPoolingLayer());

        this.layers.add(new FullyConnectedLayer(100));                   //# of hidden units
        this.layers.add(new FullyConnectedLayer(100));                   //# of hidden units
        this.layers.add(outputLayer = new FCOutputLayer(6));            //# of outputs (categories)

        setup();
    }



    public void train() {
        int maxEpoch = 1000;
        trainCurve = new ArrayList<>();
        tuneCurve = new ArrayList<>();
        double tuneAcc = 0;
        double prevTuneAcc = 0;
        int tuneAccDecreaseCnt = 0;
        int maxTuneAccDecreaseCnt = 5;

        boolean earlyStopped = false;
        for(int epoch = 0; epoch < maxEpoch && !earlyStopped ; epoch ++) {
            System.out.println("Epoch #" + epoch);
            int classNum = 6;
            int[][] confusionMatrix = new int[classNum][classNum];
            double acc = 0.0;

            List<Integer> imageIndex = new ArrayList<>();
            for(int i = 0; i < this.trainset.size(); i ++) {
                imageIndex.add(i);
            }
            Collections.shuffle(this.trainset);
//            Collections.shuffle(imageIndex);

            for (int i = 0; i < this.trainset.size(); i++) {
//                int index = imageIndex.get(i);
//                System.out.println("Train #" + index);
//                Record record = this.trainset.get(index);
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
            System.out.println("Train Accuracy: " + acc);
            for (int i = 0; i < confusionMatrix.length; i++) {
                for (int j = 0; j < confusionMatrix[0].length; j++) {
                    System.out.print("\t" + confusionMatrix[i][j]);
                }
                System.out.println();
            }

            System.out.println();


            //Early Stopping
            tuneAcc = test(this.tuneset, true);
            System.out.println("Tune Accuracy: " + tuneAcc);
            if(epoch > 20) {
                if(tuneAcc <= prevTuneAcc) {
                    tuneAccDecreaseCnt ++;
                    if( tuneAccDecreaseCnt >= maxTuneAccDecreaseCnt) {
//                        earlyStopped = true;
                    }
                }
                else {
                    tuneAccDecreaseCnt = 0;
                }
                prevTuneAcc = tuneAcc;
            }

            double testAcc = test(this.testset, true);
            System.out.println("Test Accuracy: " + testAcc);

            trainCurve.add(acc);
            tuneCurve.add(tuneAcc);
        }
        double testAcc = test(this.testset, true);
        System.out.println("Test set Accuracy: " + testAcc);

        PlotUtil plotUtil = new PlotUtil("Learning Curve");
        plotUtil.plot("Training and Tuning Set Learning Curve", trainCurve, tuneCurve);

    }

    public double test(List<Record> ds, boolean printResult) {
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
        for(int i = 0; i < 6; i ++) {
            System.out.println( outputLayer.getOutputMap(i)[0][0] );
        }
        acc /= ds.size();
        if(printResult) {
            System.out.println("Confusion Matrix: ");
            for (int i = 0; i < confusionMatrix.length; i++) {
                for (int j = 0; j < confusionMatrix[0].length; j++) {
                    System.out.print("\t" + confusionMatrix[i][j]);
                }
                System.out.println();
            }
        }
        return acc;
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
        layers.get(0).setOutputMapsNum(3);

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
                }
                else if(curLayer instanceof MaxPoolingLayer) {
                    curLayer.setOutputMapsNum(inputMapsNum);
                    curLayer.setOutputMapSize(prevLayer.getOutputMapSize().plus(1).divide(2));
                    curLayer.setKernelSize(prevLayer.getOutputMapSize());
                    curLayer.initKernels(inputMapsNum);
                }
                else {
                    throw new RuntimeException("ERROR: setup not implemented for " + curLayer.getClass().getSimpleName());
                }
            }
            curLayer.initOutputMaps();
            curLayer.initBias();
            curLayer.initErrors();
        }
    }


}
