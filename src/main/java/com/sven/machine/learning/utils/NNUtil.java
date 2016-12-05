package com.sven.machine.learning.utils;

import com.sven.machine.learning.model.Matrix;

public class NNUtil
{

	public static Matrix<Integer> getMapsize(Matrix<Integer> mapSize, Matrix<Integer> pad, Matrix<Integer> stride,
			Matrix<Integer> kernelSize)
	{
		return new Matrix<Integer>(

				getSize(mapSize.x, pad.x, stride.x, kernelSize.x), getSize(mapSize.y, pad.y, stride.y, kernelSize.y));
	}

	public static int getSize(int size, int pad, int stride, int kernelSize)
	{
		if (((size + pad * 2 - kernelSize) % stride) != 0)
		{

			throw new RuntimeException("Size can't be even divide with stride");
		}
		return (size + pad * 2 - kernelSize) / stride + 1;
	}
}
