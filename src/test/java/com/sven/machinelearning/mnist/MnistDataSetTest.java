package com.sven.machinelearning.mnist;

import org.junit.Test;

import com.sven.machine.learning.mnist.MnistDataSet;

public class MnistDataSetTest {

	@Test
	public void test() {

		MnistDataSet dataSet = new MnistDataSet("D:\\workspace\\machine-learning\\src\\main\\resources\\mnist\\train-labels.idx1-ubyte", 
				"D:\\workspace\\machine-learning\\src\\main\\resources\\mnist\\train-images.idx3-ubyte");
		int i = 0;
		while (dataSet.hasNext()) {
			System.out.println(i++ + " " + dataSet.read());
		}
	}
}
