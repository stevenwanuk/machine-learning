package com.sven.machine.learning.utils;

import com.sven.machine.learning.mnist.MnistData;

public class MathUtil
{
    
	public static double sum(double[][] error)
	{
		int m = error.length;
		int n = error[0].length;
		double sum = 0.0;
		for (int i = 0; i < m; i++)
		{
			for (int j = 0; j < n; j++)
			{
				sum += error[i][j];
			}
		}
		return sum;
	}

	public static int getMaxIndex(double[] out)
	{
		double max = out[0];
		int index = 0;
		for (int i = 1; i < out.length; i++)
			if (out[i] > max)
			{
				max = out[i];
				index = i;
			}
		return index;
	}

	public static double error(double expected, double actual)
	{
		return actual * (1 - actual) * (expected - actual);
	    //return (expected - actual) * (expected - actual)  / 2;
	}

    public static void normalizeForSigmoid(MnistData data)
    {

        double[][] d = data.getImageByte();
        for (int i = 0; i < d.length; i++)
        {
            for (int j = 0; j < d[0].length; j++)
            {

                // d[i][j] = (double) Math.round(d[i][j] / 255f * 100) / 100;
                d[i][j] = d[i][j] > 30 ? 1 : 0;
            }

        }
    }
    
    public static void normalizeForRelu(MnistData data)
    {

        double[][] d = data.getImageByte();
        for (int i = 0; i < d.length; i++)
        {
            for (int j = 0; j < d[0].length; j++)
            {

               
                d[i][j] = d[i][j] > 30 ? 1 : -1;
            }

        }
    }
	public static double sigmod(double x)
	{
		return 1 / (1 + Math.pow(Math.E, -x));
	    //return Math.max(0, x);
	}

	public interface Operator
	{
		public double process(double value);
	}

	public static final Operator one_value = new Operator()
	{

		@Override
		public double process(double value)
		{
			return 1 - value;
		}

	};

	public static final Operator digmod = new Operator()
	{

		@Override
		public double process(double value)
		{
			return 1 / (1 + Math.pow(Math.E, -value));
		}

	};

	interface OperatorOnTwo
	{
		public double process(double a, double b);
	}

	public static final OperatorOnTwo plus = new OperatorOnTwo()
	{

		@Override
		public double process(double a, double b)
		{
			return a + b;
		}

	};

	public static OperatorOnTwo multiply = new OperatorOnTwo()
	{

		@Override
		public double process(double a, double b)
		{
			return a * b;
		}

	};

	public static OperatorOnTwo minus = new OperatorOnTwo()
	{

		@Override
		public double process(double a, double b)
		{
			return a - b;
		}

	};
}
