package com.sven.machine.learning.network;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.sven.machine.learning.layer.Layer;
import com.sven.machine.learning.mnist.MnistData;

public class CNN
{
	Logger log = LoggerFactory.getLogger(this.getClass());

	private CNNConfig config;

	public CNN(CNNConfig config)
	{
		this.config = config;
	}

	public void init()
	{

		Layer prevLayer = null;
		int index = 0;
		for (Layer layer : config.getLayers())
		{

			layer.setLayerIndex(index);
			if (prevLayer != null)
			{
				prevLayer.setAfterLayer(layer);
				layer.setPrevLayer(prevLayer);
			}
			layer.init();
			log.debug("init layer:" + index + " " + layer.getLayerType() + " " + layer.getMapSize() + " "
					+ layer.getMapNumber());

			prevLayer = layer;
			index++;
		}
	}

	protected void train(MnistData data)
	{
		forward(data);
		bp(data);
		learning();
		// boolean result = backPropagation(data);
	}

	private void forward(MnistData record)
	{
		config.getLayers().forEach(layer ->
		{
			layer.forward(record);
		});
	}

	private void bp(MnistData record)
	{
		for (int i = 0; i < config.getLayers().size(); i++)
		{

			config.getLayers().get(config.getLayers().size() - 1 - i).bp(record);
		}
	}

	private void learning()
	{
		config.getLayers().forEach(layer -> layer.learnFromErrors());
	}

	protected boolean backPropagation(MnistData record)
	{
		// boolean result = setOutLayerErrors(record);
		// setHiddenLayerErrors();
		// return result;
		return false;
	}

	public void verify(MnistData data)
	{

	}

	public CNNConfig getConfig()
	{
		return config;
	}

	public void setConfig(CNNConfig config)
	{
		this.config = config;
	}

}
