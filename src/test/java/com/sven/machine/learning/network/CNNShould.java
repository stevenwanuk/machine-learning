package com.sven.machine.learning.network;

import java.util.ArrayList;
import java.util.List;

import com.sven.machine.learning.data.DataSet;
import com.sven.machine.learning.data.DataSet.Record;
import com.sven.machine.learning.enums.ActivationType;
import com.sven.machine.learning.enums.LostFunctionType;
import com.sven.machine.learning.enums.PoolingType;
import com.sven.machine.learning.layer.ConvolutionalLayer;
import com.sven.machine.learning.layer.InputLayer;
import com.sven.machine.learning.layer.Layer;
import com.sven.machine.learning.layer.OutputLayer;
import com.sven.machine.learning.layer.SubsamplingLayer;
import com.sven.machine.learning.mnist.MnistData;
import com.sven.machine.learning.mnist.MnistDataSet;
import com.sven.machine.model.Matrix;

public class CNNShould
{

	public static MnistDataSet readTrainData()
	{

		MnistDataSet dataSet = new MnistDataSet(
				"D:\\workspace\\machine-learning\\src\\main\\resources\\mnist\\train-labels.idx1-ubyte",
				"D:\\workspace\\machine-learning\\src\\main\\resources\\mnist\\train-images.idx3-ubyte");
		return dataSet;
	}

	public static CNNConfig buildNetwork()
	{
		CNNConfig config = new CNNConfig();
		config.setLearningRate(0.01);
		config.setLayers(buildLayers());
		return config;
	}

	public static List<Layer> buildLayers()
	{

		List<Layer> layers = new ArrayList<>();

		InputLayer i0 = new InputLayer();
		i0.setMapSize(new Matrix<Integer>(28, 28));
		layers.add(i0);

		// 28*28
		ConvolutionalLayer c0 = new ConvolutionalLayer();
		c0.setKernelSize(new Matrix<>(5, 5));
		c0.setPadding(new Matrix<>(0, 0));
		c0.setStride(new Matrix<>(1, 1));
		c0.setMapNumber(20);
		c0.setChannelSize(1);
		layers.add(c0);
		// 24*24*20

		SubsamplingLayer s0 = new SubsamplingLayer();
		s0.setKernelSize(new Matrix<>(2, 2));
		s0.setStride(new Matrix<>(2, 2));
		s0.setPadding(new Matrix<>(0, 0));
		s0.setPoolingType(PoolingType.max);
		layers.add(s0);
		// 12*12*20
		ConvolutionalLayer c1 = new ConvolutionalLayer();
		c1.setKernelSize(new Matrix<>(5, 5));
		c1.setPadding(new Matrix<>(0, 0));
		c1.setStride(new Matrix<>(1, 1));
		c1.setMapNumber(50);
		layers.add(c1);
		// 8*8*50
		SubsamplingLayer s1 = new SubsamplingLayer();
		s1.setKernelSize(new Matrix<>(2, 2));
		s1.setStride(new Matrix<>(2, 2));
		s1.setPadding(new Matrix<>(0, 0));
		s1.setPoolingType(PoolingType.max);
		layers.add(s1);
		// 4*4*50
		// DenseLayer d0 = new DenseLayer();
		// d0.setOutputNumber(500);
		// d0.setActivationType(ActivationType.relu);
		// layers.add(s0);

		// 500
		OutputLayer o0 = new OutputLayer();
		o0.setLostFunctionType(LostFunctionType.negativeLogLikelihood);
		o0.setOutputNumber(10);
		o0.setActivationType(ActivationType.softmax);
		layers.add(o0);

		return layers;
	}

	public static void main(String[] args)
	{

		// MnistDataSet trainData = readTrainData();
		// trainData.read();

		CNN cnn = new CNN(buildNetwork());
		cnn.init();
		// MnistDataSet dataSet = readTrainData();
		// MnistData data = dataSet.read();
		// while (data != null)
		// {
		// cnn.train(data);
		// data = dataSet.read();
		// }

		for (int i = 0; i < 784; i++)
		{

			Record record = DataSet.load("D:\\workspace\\cnn\\mnist\\train.format", ",", 784).getRecord(i);
			MnistData data = new MnistData();
			data.setLabel(record.getLabel().intValue());

			int[][] temp = new int[28][28];
			for (int x = 0; x < 28; x++)
			{
				for (int y = 0; y < 28; y++)
				{
					temp[x][y] = (int) (record.getAttrs()[x * 28 + y]);
				}
			}
			data.setImageByte(temp);
			cnn.train(data);
		}

	}
}
