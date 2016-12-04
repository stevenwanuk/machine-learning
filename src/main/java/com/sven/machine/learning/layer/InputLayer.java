package com.sven.machine.learning.layer;

import com.sven.machine.learning.enums.LayerType;
import com.sven.machine.learning.mnist.MnistData;

public class InputLayer extends Layer
{
	public InputLayer()
	{

		this.setLayerType(LayerType.inputLayer);
	}

	@Override
	public void forward(MnistData mnistData)
	{
		int[][] data = mnistData.getImageByte();
		for (int i = 0; i < mapSize.x; i++)
		{
			for (int j = 0; j < mapSize.y; j++)
			{
				maps[0][i][j] = data[i][j];
			}
		}

	}

	@Override
	public void init()
	{
		this.setMapNumber(1);
		maps = new double[1][mapSize.x][mapSize.y];
		bias = new double[mapNumber];
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
