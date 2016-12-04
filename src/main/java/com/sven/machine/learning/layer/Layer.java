package com.sven.machine.learning.layer;

import com.sven.machine.learning.enums.LayerType;
import com.sven.machine.learning.mnist.MnistData;
import com.sven.machine.learning.utils.MathUtil.Operator;
import com.sven.machine.model.Matrix;

public abstract class Layer
{

	protected Layer prevLayer;
	protected Layer nextLayer;
	protected int layerIndex;
	protected LayerType layerType;

	protected Matrix<Integer> mapSize;
	protected int mapNumber;

	protected Matrix<Integer> kernelSize;
	protected double[][][][] kernel;

	protected double[][][] maps;
	protected double[] bias;
	protected double[][][] errors;

	abstract public void init();

	abstract public void forward(MnistData data);

	abstract public void bp(MnistData data);

	abstract public void learnFromErrors();

	protected static double ALPHA = 0.85;
	protected static final double LAMBDA = 0;

	protected Operator multiply_alpha = new Operator()
	{

		@Override
		public double process(double value)
		{

			return value * ALPHA;
		}

	};

	protected Operator multiply_lambda = new Operator()
	{

		@Override
		public double process(double value)
		{

			return value * (1 - LAMBDA * ALPHA);
		}

	};

	public Layer getNextLayer()
	{
		return nextLayer;
	}

	public void setNextLayer(Layer nextLayer)
	{
		this.nextLayer = nextLayer;
	}

	public Matrix<Integer> getKernelSize()
	{
		return kernelSize;
	}

	public void setKernelSize(Matrix<Integer> kernelSize)
	{
		this.kernelSize = kernelSize;
	}

	public double[][][][] getKernel()
	{
		return kernel;
	}

	public void setKernel(double[][][][] kernel)
	{
		this.kernel = kernel;
	}

	public double[][][] getErrors()
	{
		return errors;
	}

	public void setErrors(double[][][] errors)
	{
		this.errors = errors;
	}

	public double[] getBias()
	{
		return bias;
	}

	public void setBias(double[] bias)
	{
		this.bias = bias;
	}

	public int getMapNumber()
	{
		return mapNumber;
	}

	public void setMapNumber(int mapNumber)
	{
		this.mapNumber = mapNumber;
	}

	public double[][][] getMaps()
	{
		return maps;
	}

	public void setMaps(double[][][] maps)
	{
		this.maps = maps;
	}

	public Matrix<Integer> getMapSize()
	{
		return mapSize;
	}

	public void setMapSize(Matrix<Integer> mapSize)
	{
		this.mapSize = mapSize;
	}

	public Layer getPrevLayer()
	{
		return prevLayer;
	}

	public void setPrevLayer(Layer prevLayer)
	{
		this.prevLayer = prevLayer;
	}

	public Layer getAfterLayer()
	{
		return nextLayer;
	}

	public void setAfterLayer(Layer afterLayer)
	{
		this.nextLayer = afterLayer;
	}

	public int getLayerIndex()
	{
		return layerIndex;
	}

	public void setLayerIndex(int layerIndex)
	{
		this.layerIndex = layerIndex;
	}

	public LayerType getLayerType()
	{
		return layerType;
	}

	public void setLayerType(LayerType layerType)
	{
		this.layerType = layerType;
	}

}
