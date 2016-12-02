package com.sven.machine.learning.network;

import java.util.List;

import com.sven.machine.learning.Layer.Layer;
import com.sven.machine.learning.mnist.MnistData;

public class CNN
{

    private List<Layer> layers;
    private double learningRate;

    public List<Layer> getLayers()
    {
        return layers;
    }

    public void setLayers(List<Layer> layers)
    {
        this.layers = layers;
    }

    public double getLearningRate()
    {
        return learningRate;
    }

    public void setLearningRate(double learningRate)
    {
        this.learningRate = learningRate;
    }

    public void init()
    {

    }

    public void train(MnistData data)
    {

    }

    public void verify(MnistData data)
    {

    }
}
