package com.sven.machine.learning.layer;

import com.sven.machine.learning.enums.LayerType;
import com.sven.machine.learning.enums.PoolingType;
import com.sven.machine.learning.mnist.MnistData;
import com.sven.machine.learning.utils.MathUtil;
import com.sven.machine.learning.utils.MatrixUtil;
import com.sven.machine.learning.utils.NNUtil;
import com.sven.machine.model.Matrix;

public class SubsamplingLayer extends Layer
{
	public PoolingType poolingType;
	private Matrix<Integer> stride;
	private Matrix<Integer> padding;

	public SubsamplingLayer()
	{
		this.setLayerType(LayerType.subsamplingLayer);
	}

	@Override
	public void init()
	{
		this.mapSize = NNUtil.getMapsize(prevLayer.getMapSize(), padding, stride, kernelSize);

		this.mapNumber = prevLayer.mapNumber;
		errors = new double[mapNumber][mapSize.x][mapSize.y];
		this.maps = new double[mapNumber][mapSize.x][mapSize.y];
	}

	@Override
	public void forward(MnistData data)
	{
		double[][][] prevMaps = prevLayer.getMaps();
		for (int i = 0; i < prevMaps.length; i++)
		{

			maps[i] = MatrixUtil.scaleMatrix(prevMaps[i], this.stride);
		}

	}

	public PoolingType getPoolingType()
	{
		return poolingType;
	}

	public void setPoolingType(PoolingType poolingType)
	{
		this.poolingType = poolingType;
	}

	public Matrix<Integer> getKernelSize()
	{
		return kernelSize;
	}

	public void setKernelSize(Matrix<Integer> kernelSize)
	{
		this.kernelSize = kernelSize;
	}

	public Matrix<Integer> getStride()
	{
		return stride;
	}

	public void setStride(Matrix<Integer> stride)
	{
		this.stride = stride;
	}

	public Matrix<Integer> getPadding()
	{
		return padding;
	}

	public void setPadding(Matrix<Integer> padding)
	{
		this.padding = padding;
	}

	@Override
	public void bp(MnistData data)
	{
		for (int i = 0; i < mapNumber; i++)
		{
			double[][] sum = null;
			for (int j = 0; j < nextLayer.mapNumber; j++)
			{
				double[][] nextError = nextLayer.errors[j];
				double[][] kernel = nextLayer.kernel[i][j];

				sum = MatrixUtil.matrixOp(MatrixUtil.convnFull(nextError, MatrixUtil.rot180(kernel)), sum, null, null,
						MathUtil.plus);
			}
			errors[i] = sum;
		}

	}

	@Override
	public void learnFromErrors()
	{
		// TODO Auto-generated method stub

	}

}
