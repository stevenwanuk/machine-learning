package com.sven.machine.learning.network;

import java.util.List;

import com.sven.machine.learning.layer.Layer;

public class CNNConfig
{
	private List<Layer> layers;
	private double learningRate;
	private int batchSize;
	private int epoch;

	public int getBatchSize()
	{
		return batchSize;
	}

	public void setBatchSize(int batchSize)
	{
		this.batchSize = batchSize;
	}

	public int getEpoch()
	{
		return epoch;
	}

	public void setEpoch(int epoch)
	{
		this.epoch = epoch;
	}

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

}
