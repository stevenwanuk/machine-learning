package com.sven.machine.learning.layer;

import com.sven.machine.learning.enums.LayerType;
import com.sven.machine.learning.mnist.MnistData;
import com.sven.machine.learning.utils.MathUtil;
import com.sven.machine.learning.utils.MatrixUtil;
import com.sven.machine.learning.utils.NNUtil;
import com.sven.machine.model.Matrix;

public class ConvolutionalLayer extends Layer
{

	private int channelSize;
	private Matrix<Integer> stride;
	private Matrix<Integer> padding;

	@Override
	public void init()
	{
		Matrix<Integer> convolutionalMapSize = NNUtil.getMapsize(prevLayer.getMapSize(), padding, stride, kernelSize);
		mapSize = convolutionalMapSize;
		maps = new double[mapNumber][convolutionalMapSize.x][convolutionalMapSize.y];
		errors = new double[mapNumber][mapSize.x][mapSize.y];

		kernel = new double[prevLayer.getMapNumber()][mapNumber][kernelSize.x][kernelSize.y];
		for (int i = 0; i < prevLayer.getMapNumber(); i++)
		{
			for (int j = 0; j < mapNumber; j++)
			{
				kernel[i][j] = MatrixUtil.randomMatrix(kernelSize.x, kernelSize.y);
			}
		}

		bias = MatrixUtil.randomArray(mapNumber);

	}

	@Override
	public void forward(MnistData data)
	{
		int prevMapNumber = prevLayer.getMapNumber();
		for (int mapIndex = 0; mapIndex < mapNumber; mapIndex++)
		{
			double[][] sum = null;
			for (int prevMapIndex = 0; prevMapIndex < prevMapNumber; prevMapIndex++)
			{
				// Something wrong here. seems not to sum
				double[][] prevMap = prevLayer.getMaps()[prevMapIndex];

				sum = MatrixUtil.matrixOp(
						MatrixUtil.convnValid(prevMap, kernel[prevMapIndex][mapIndex], padding, stride), sum, null,
						null, MathUtil.plus);
			}

			final double mapBias = bias[mapIndex];
			maps[mapIndex] = MatrixUtil.matrixOp(sum, new MathUtil.Operator()
			{

				@Override
				public double process(double value)
				{
					return MathUtil.sigmod(value + mapBias);
				}

			});
		}
	}

	public ConvolutionalLayer()
	{
		this.setLayerType(LayerType.convolutionalLayer);
	}

	public int getChannelSize()
	{
		return channelSize;
	}

	public void setChannelSize(int channelSize)
	{
		this.channelSize = channelSize;
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
			SubsamplingLayer subsamplingLayer = (SubsamplingLayer) nextLayer;

			Matrix<Integer> scale = subsamplingLayer.getKernelSize();
			double[][] nextError = nextLayer.errors[i];
			double[][] map = maps[i];
			double[][] outMatrix = MatrixUtil.matrixOp(map, MatrixUtil.cloneMatrix(map), null, MathUtil.one_value,
					MathUtil.multiply);
			errors[i] = MatrixUtil.matrixOp(outMatrix, MatrixUtil.kronecker(nextError, scale), null, null,
					MathUtil.multiply);
		}

	}

	@Override
	public void learnFromErrors()
	{
		// updateKernels
		for (int i = 0; i < mapNumber; i++)
		{
			for (int j = 0; j < prevLayer.mapNumber; j++)
			{
				double[][] deltaKernel = null;
				double[][] sum = MatrixUtil.matrixOp(MatrixUtil.convnValid(prevLayer.maps[j], errors[i]), deltaKernel,
						null, null, MathUtil.plus);
				kernel[j][i] = MatrixUtil.matrixOp(kernel[j][i], sum, multiply_lambda, multiply_alpha, MathUtil.plus);
			}
		}
		// updateBias
		for (int i = 0; i < mapNumber; i++)
		{

			bias[i] += ALPHA * MathUtil.sum(errors[i]);
		}
	}

}
