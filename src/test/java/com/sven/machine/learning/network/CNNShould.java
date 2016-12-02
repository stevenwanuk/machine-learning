package com.sven.machine.learning.network;

import java.util.ArrayList;
import java.util.List;

import com.sven.machine.learning.Layer.ConvolutionLayer;
import com.sven.machine.learning.Layer.DenseLayer;
import com.sven.machine.learning.Layer.Layer;
import com.sven.machine.learning.Layer.OutputLayer;
import com.sven.machine.learning.Layer.SubsampleLayer;
import com.sven.machine.learning.enums.ActivationType;
import com.sven.machine.learning.enums.LostFunctionType;
import com.sven.machine.learning.enums.PoolingType;
import com.sven.machine.learning.mnist.MnistDataSet;
import com.sven.machine.model.Matrix;

public class CNNShould
{

    public static MnistDataSet readTrainData()
    {

        MnistDataSet dataSet = new MnistDataSet(
                "C:\\workspace\\machineLearning\\machine-learning-example\\src\\main\\resources\\mnist\\train-labels.idx1-ubyte",
                "C:\\workspace\\machineLearning\\machine-learning-example\\src\\main\\resources\\mnist\\train-images.idx3-ubyte");
        return dataSet;
    }

    public static CNN buildNetwork()
    {
        CNN cnn = new CNN();
        cnn.setLearningRate(0.01);
        cnn.setLayers(buildLayers());
        return cnn;
    }

    public static List<Layer> buildLayers()
    {
        
        List<Layer> layers = new ArrayList<>();
        //28*28
        ConvolutionLayer c0 = new ConvolutionLayer();
        c0.setKernelSize(new Matrix<>(5, 5));
        c0.setPadding(new Matrix<>(0, 0));
        c0.setStride(new Matrix<>(1, 1));
        c0.setFilterSize(20);
        c0.setChannelSize(1);
        layers.add(c0);
        //24*24*20
        
        SubsampleLayer s0 = new SubsampleLayer();
        s0.setKernelSize(new Matrix<>(2, 2));
        s0.setStride(new Matrix<>(2, 2));
        s0.setPoolingType(PoolingType.max);
        layers.add(s0);
        //12*12*20
        ConvolutionLayer c1 = new ConvolutionLayer();
        c1.setKernelSize(new Matrix<>(5, 5));
        c1.setPadding(new Matrix<>(0, 0));
        c1.setStride(new Matrix<>(1, 1));
        c1.setFilterSize(50);
        layers.add(c1);
        //8*8*1000
        SubsampleLayer s1 = new SubsampleLayer();
        s1.setKernelSize(new Matrix<>(2, 2));
        s1.setStride(new Matrix<>(2, 2));
        s1.setPoolingType(PoolingType.max);
        layers.add(s1);
        //4*4*1000
        DenseLayer d0 = new DenseLayer();
        d0.setOutputNumber(500);
        d0.setActivationType(ActivationType.relu);
        layers.add(s0);
        
        //500
        OutputLayer o0 = new OutputLayer();
        o0.setLostFunctionType(LostFunctionType.negativeLogLikelihood);
        o0.setOutputNumber(10);
        o0.setActivationType(ActivationType.softmax);
        layers.add(o0);
        
        return layers;
    }

    public static void main(String[] args)
    {

        MnistDataSet trainData = readTrainData();
        trainData.read();
        
        CNN cnn = buildNetwork();

    }
}
