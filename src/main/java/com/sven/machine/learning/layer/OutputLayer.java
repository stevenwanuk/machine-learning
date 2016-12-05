package com.sven.machine.learning.layer;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.sven.machine.learning.enums.ActivationType;
import com.sven.machine.learning.enums.LayerType;
import com.sven.machine.learning.enums.LostFunctionType;
import com.sven.machine.learning.mnist.MnistData;
import com.sven.machine.learning.model.Matrix;
import com.sven.machine.learning.utils.MathUtil;
import com.sven.machine.learning.utils.MatrixUtil;

public class OutputLayer extends Layer
{

	Logger log = LoggerFactory.getLogger(this.getClass());
	private int outputNumber;
	private ActivationType activationType;
	private LostFunctionType lostFunctionType;

	@Override
	public void init()
	{

		bias = MatrixUtil.randomArray(outputNumber);
		mapSize = new Matrix<Integer>(1, 1);
		mapNumber = outputNumber;
		maps = new double[outputNumber][1][1];
		errors = new double[mapNumber][mapSize.x][mapSize.y];
		kernel = new double[prevLayer.getMapNumber()][mapNumber][prevLayer.getMapSize().x][prevLayer.getMapSize().y];
		for (int i = 0; i < prevLayer.getMapNumber(); i++)
		{
			for (int j = 0; j < mapNumber; j++)
			{
				kernel[i][j] = MatrixUtil.randomMatrix(prevLayer.getMapSize().x, prevLayer.getMapSize().y);
			}
		}
	}

	@Override
	public void forward(MnistData data)
	{
		int prevMapNumber = prevLayer.getMapNumber();
		for (int mapIndex = 0; mapIndex < outputNumber; mapIndex++)
		{
			double[][] sum = null;
			for (int prevMapIndex = 0; prevMapIndex < prevMapNumber; prevMapIndex++)
			{
				// Something wrong here. seems not to sum
				double[][] prevMap = prevLayer.getMaps()[prevMapIndex];

				sum = MatrixUtil.matrixOp(MatrixUtil.convnValid(prevMap, kernel[prevMapIndex][mapIndex], null, null),
						sum, null, null, MathUtil.plus);
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

	@Override
	public void bp(MnistData data)
	{
		int label = data.getLabel();
		double[] target = new double[mapNumber];
		target[label] = 1;

		double[] temp = new double[mapNumber];
		for (int i = 0; i < mapNumber; i++)
		{
			temp[i] = maps[i][0][0];
		}

		log.info("expected:" + label + ", actual:" + MathUtil.getMaxIndex(temp));

		for (int i = 0; i < mapNumber; i++)
		{
			errors[i][0][0] = MathUtil.error(target[i], maps[i][0][0]);
		}

	}

	public OutputLayer()
	{
		this.setLayerType(LayerType.outputLayer);
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

	public LostFunctionType getLostFunctionType()
	{
		return lostFunctionType;
	}

	public void setLostFunctionType(LostFunctionType lostFunctionType)
	{
		this.lostFunctionType = lostFunctionType;
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
