package com.sven.machine.learning.layer;

import com.sven.machine.learning.enums.ActivationType;
import com.sven.machine.learning.enums.LayerType;
import com.sven.machine.learning.mnist.MnistData;

public class DenseLayer extends Layer
{

	private int outputNumber;
	private ActivationType activationType;

	public DenseLayer()
	{
		this.setLayerType(LayerType.denseLayer);
	}

	public int getOutputNumber()
	{
		return outputNumber;
	}

	public void setOutputNumber(int outputNumber)
	{
		this.outputNumber = outputNumber;
	}

	public ActivationType getActivationType()
	{
		return activationType;
	}

	public void setActivationType(ActivationType activationType)
	{
		this.activationType = activationType;
	}

	@Override
	public void init()
	{
		// TODO Auto-generated method stub

	}

	@Override
	public void forward(MnistData data)
	{
		// TODO Auto-generated method stub

	}

	@Override
	public void bp(MnistData data)
	{
		// TODO Auto-generated method stub

	}

	@Override
	public void learnFromErrors()
	{
		// TODO Auto-generated method stub

	}

}
