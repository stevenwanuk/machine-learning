package com.sven.machine.learning.utils;

import java.util.Random;

import com.sven.machine.learning.model.Matrix;
import com.sven.machine.learning.utils.MathUtil.Operator;
import com.sven.machine.learning.utils.MathUtil.OperatorOnTwo;

public class MatrixUtil
{
	public static double[][] convnFull(double[][] matrix, final double[][] kernel)
	{
		int m = matrix.length;
		int n = matrix[0].length;
		final int km = kernel.length;
		final int kn = kernel[0].length;
		final double[][] extendMatrix = new double[m + 2 * (km - 1)][n + 2 * (kn - 1)];
		for (int i = 0; i < m; i++)
		{
			for (int j = 0; j < n; j++)
				extendMatrix[i + km - 1][j + kn - 1] = matrix[i][j];
		}
		return convnValid(extendMatrix, kernel);
	}

	public static double sumWithStartIndex(final double[][] map, double[][] kernel, Matrix<Integer> startMatrix)
	{
		int kx = kernel.length;
		int ky = kernel[0].length;

		int sx = startMatrix.x;
		int sy = startMatrix.y;

		double sum = 0;
		for (int i = 0; i < kx; i++)
		{
			for (int j = 0; j < ky; j++)
			{
				sum += map[i + sx][j + sy] * kernel[i][j];
			}
		}
		return sum;
	}

	public static double[][] convnValid(final double[][] map, double[][] kernel)
	{

		return convnValid(map, kernel, new Matrix<Integer>(1, 1));
	}

	public static double[][] convnValid(final double[][] map, double[][] kernel, Matrix<Integer> stride)
	{
		int x = map.length;
		int y = map[0].length;

		int kx = kernel.length;
		int ky = kernel[0].length;

		int sx = stride.x;
		int sy = stride.y;

		if ((x - kx) % sx != 0 || (y - ky) % sy != 0)
		{
			throw new RuntimeException("the map is not even");
		}
		int mx = (x - kx) / sx + 1;
		int my = (y - ky) / sy + 1;

		double[][] result = new double[mx][my];
		for (int i = 0; i < mx; i++)
		{
			for (int j = 0; j < my; j++)
			{

				result[i][j] = sumWithStartIndex(map, kernel, new Matrix<Integer>(i * sx, j * sy));
			}
		}
		return result;

	}

	public static double[][] convnValid(final double[][] map, double[][] kernel, Matrix<Integer> padding,
			Matrix<Integer> stride)
	{

		if (padding == null)
		{
			padding = new Matrix<Integer>(0, 0);
		}
		if (stride == null)
		{
			stride = new Matrix<Integer>(1, 1);
		}

		// kernel = rot180(kernel);
		int x = map.length;
		int y = map[0].length;

		int px = padding.x;
		int py = padding.y;

		int nx = x + 2 * px;
		int ny = y + 2 * py;
		double[][] newMap = new double[nx][ny];
		for (int i = 0; i < x; i++)
		{
			for (int j = 0; j < y; j++)
			{
				newMap[i + px][j + py] = map[i][j];

			}
		}
		return convnValid(newMap, kernel, stride);

	}

	public static double[][] matrixOp(final double[][] ma, final double[][] mb, final Operator operatorA,
			final Operator operatorB, OperatorOnTwo operator)
	{
		if (ma == null)
		{
			return mb;
		}
		if (mb == null)
		{
			return ma;
		}

		final int m = ma.length;
		int n = ma[0].length;
		if (m != mb.length || n != mb[0].length)
			throw new RuntimeException("ma.length:" + ma.length + "  mb.length:" + mb.length);

		for (int i = 0; i < m; i++)
		{
			for (int j = 0; j < n; j++)
			{
				double a = ma[i][j];
				if (operatorA != null)
					a = operatorA.process(a);
				double b = mb[i][j];
				if (operatorB != null)
					b = operatorB.process(b);
				mb[i][j] = operator.process(a, b);
			}
		}
		return mb;
	}

	public static double[][] scaleMatrix(final double[][] matrix, final Matrix<Integer> scale)
	{
		int m = matrix.length;
		int n = matrix[0].length;
		final int sm = m / scale.x;
		final int sn = n / scale.y;
		final double[][] outMatrix = new double[sm][sn];
		if (sm * scale.x != m || sn * scale.y != n)
			throw new RuntimeException("scale matrix");
		final int size = scale.x * scale.y;
		for (int i = 0; i < sm; i++)
		{
			for (int j = 0; j < sn; j++)
			{
				double sum = 0.0;
				for (int si = i * scale.x; si < (i + 1) * scale.x; si++)
				{
					for (int sj = j * scale.y; sj < (j + 1) * scale.y; sj++)
					{
						sum += matrix[si][sj];
					}
				}
				outMatrix[i][j] = sum / size;
			}
		}
		return outMatrix;
	}

	public static double[][] matrixOp(final double[][] ma, Operator operator)
	{
		final int m = ma.length;
		int n = ma[0].length;
		for (int i = 0; i < m; i++)
		{
			for (int j = 0; j < n; j++)
			{
				ma[i][j] = operator.process(ma[i][j]);
			}
		}
		return ma;
	}

	private static Random r = new Random(2);

	public static double[] randomArray(int len)
	{
		double[] data = new double[len];
		for (int i = 0; i < len; i++)
		{
			// data[i] = r.nextDouble() / 10 - 0.05;
			data[i] = 0;
		}
		return data;
	}
	
	   public static double[] initArray(int len, double initWeight)
	    {
	        double[] data = new double[len];
	        for (int i = 0; i < len; i++)
	        {
	            // data[i] = r.nextDouble() / 10 - 0.05;
	            data[i] = initWeight;
	        }
	        return data;
	    }

	public static double[][] rot180(double[][] matrix)
	{
		matrix = cloneMatrix(matrix);
		int m = matrix.length;
		int n = matrix[0].length;
		for (int i = 0; i < m; i++)
		{
			for (int j = 0; j < n / 2; j++)
			{
				double tmp = matrix[i][j];
				matrix[i][j] = matrix[i][n - 1 - j];
				matrix[i][n - 1 - j] = tmp;
			}
		}
		for (int j = 0; j < n; j++)
		{
			for (int i = 0; i < m / 2; i++)
			{
				double tmp = matrix[i][j];
				matrix[i][j] = matrix[m - 1 - i][j];
				matrix[m - 1 - i][j] = tmp;
			}
		}
		return matrix;
	}

	public static double[][] kronecker(final double[][] matrix, final Matrix<Integer> scale)
	{
		final int m = matrix.length;
		int n = matrix[0].length;
		final double[][] outMatrix = new double[m * scale.x][n * scale.y];

		for (int i = 0; i < m; i++)
		{
			for (int j = 0; j < n; j++)
			{
				for (int ki = i * scale.x; ki < (i + 1) * scale.x; ki++)
				{
					for (int kj = j * scale.y; kj < (j + 1) * scale.y; kj++)
					{
						outMatrix[ki][kj] = matrix[i][j];
					}
				}
			}
		}
		return outMatrix;
	}

	public static double[][] cloneMatrix(final double[][] matrix)
	{
		final int m = matrix.length;
		int n = matrix[0].length;
		final double[][] outMatrix = new double[m][n];

		for (int i = 0; i < m; i++)
		{
			for (int j = 0; j < n; j++)
			{
				outMatrix[i][j] = matrix[i][j];
			}
		}
		return outMatrix;
	}

	
	public static double[][] initMatrix(int x, int y, double initWeight) 
	{
	    double[][] matrix = new double[x][y];
	    for (int i = 0; i < x; i++)
        {
            for (int j = 0; j < y; j++)
            {

                matrix[i][j] = initWeight;
            }
        }
	    return matrix;
	}
	
	public static double[][] randomMatrix(int x, int y)
	{
		double[][] matrix = new double[x][y];
		// int tag = 1;
		for (int i = 0; i < x; i++)
		{
			for (int j = 0; j < y; j++)
			{
				// [-0.05,0.05)
				matrix[i][j] = (r.nextDouble() - 0.05) / 10;
				// matrix[i][j] = tag * 0.5;
				// if (b)
				// matrix[i][j] *= 1.0*(i + j + 2) / (i + 1) / (j + 1);
				// tag *= -1;
			}
		}
		// printMatrix(matrix);
		return matrix;
	}
}
