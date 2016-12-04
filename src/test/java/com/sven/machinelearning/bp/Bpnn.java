package com.sven.machinelearning.bp;

import java.util.Random;

public class Bpnn {

	private double[] input;

	private double[] hiddenLayer;
	private double[] hiddenDelta;
	public double hidErrSum = 0d;

	private double[] output;
	private double[] outputDelta;
	public double optErrSum = 0d;

	private double learningRate;
	private double momentum;

	private final double[] target;

	private final Random random = new Random(19881211);
	/**
	 * weight matrix from input layer to hidden layer.
	 */
	private final double[][] iptHidWeights;
	/**
	 * weight matrix from hidden layer to output layer.
	 */

	/**
	 * previous weight update.
	 */
	private final double[][] iptHidPrevUptWeights;
	/**
	 * previous weight update.
	 */
	private final double[][] hidOptPrevUptWeights;

	private final double[][] hidOptWeights;

	public Bpnn(int inputSize, int hiddenSize, int outputSize, double learningRate, double momentum) {
		input = new double[inputSize + 1];
		hiddenLayer = new double[hiddenSize + 1];
		output = new double[outputSize + 1];
		target = new double[outputSize + 1];

		hiddenDelta = new double[hiddenSize + 1];
		outputDelta = new double[outputSize + 1];

		iptHidWeights = new double[inputSize + 1][hiddenSize + 1];
		hidOptWeights = new double[hiddenSize + 1][outputSize + 1];

		randomizeWeights(iptHidWeights);
		randomizeWeights(hidOptWeights);

		iptHidPrevUptWeights = new double[inputSize + 1][hiddenSize + 1];
		hidOptPrevUptWeights = new double[hiddenSize + 1][outputSize + 1];

		this.learningRate = learningRate;
		this.momentum = momentum;
	}

	/**
	 * Entry method. The train data should be a one-dim vector.
	 * 
	 * @param trainData
	 * @param target
	 */
	public void train(double[] trainData, double[] target) {
		loadInput(trainData);
		loadTarget(target);
		forward();
		calculateDelta();
		adjustWeight();
	}
	
	private void adjustWeight(double[] delta, double[] layer, double[][] weight, double[][] prevWeight) {

		layer[0] = 1;
		for (int i = 1, len = delta.length; i != len; ++i) {
			for (int j = 0, len2 = layer.length; j != len2; ++j) {
				double newVal = momentum * prevWeight[j][i] + learningRate * delta[i] * layer[j];
				weight[j][i] += newVal;
				prevWeight[j][i] = newVal;
			}
		}
	}
	
	private void adjustWeight() {
		adjustWeight(outputDelta, hiddenLayer, hidOptWeights, hidOptPrevUptWeights);
		adjustWeight(hiddenDelta, input, iptHidWeights, iptHidPrevUptWeights);
	}
	
	private void outputErr() {
		double errSum = 0;
		for (int idx = 1, len = outputDelta.length; idx != len; ++idx) {
			double o = output[idx];
			outputDelta[idx] = o * (1d - o) * (target[idx] - o);
			errSum += Math.abs(outputDelta[idx]);
		}
		optErrSum = errSum;
	}
	private void hiddenErr() {
		double errSum = 0;
		for (int j = 1, len = hiddenDelta.length; j != len; ++j) {
			double o = hiddenDelta[j];
			double sum = 0;
			for (int k = 1, len2 = outputDelta.length; k != len2; ++k)
				sum += hidOptWeights[j][k] * outputDelta[k];
			hiddenDelta[j] = o * (1d - o) * sum;
			errSum += Math.abs(hiddenDelta[j]);
		}
		hidErrSum = errSum;
	}
	private void calculateDelta() {
		outputErr();
		hiddenErr();
	}
	
	private void forward() {
		forward(input, hiddenLayer, iptHidWeights);
		forward(hiddenLayer, output, hidOptWeights);
	}
	
	private void forward(double[] layer0, double[] layer1, double[][] weight) {
		// threshold unit.
		layer0[0] = 1.0;
		for (int j = 1, len = layer1.length; j != len; ++j) {
			double sum = 0;
			for (int i = 0, len2 = layer0.length; i != len2; ++i)
				sum += weight[i][j] * layer0[i];
			layer1[j] = sigmoid(sum);
		}
	}

	private double sigmoid(double val) {
		return 1d / (1d + Math.exp(-val));
	}
	
	private void loadInput(double[] inData) {
		if (inData.length != input.length - 1) {
			throw new IllegalArgumentException("Size Do Not Match.");
		}
		System.arraycopy(inData, 0, input, 1, inData.length);
	}
	
	private void loadTarget(double[] arg) {
		if (arg.length != target.length - 1) {
			throw new IllegalArgumentException("Size Do Not Match.");
		}
		System.arraycopy(arg, 0, target, 1, arg.length);
	}
	
	private void randomizeWeights(double[][] matrix) {
		for (int i = 0, len = matrix.length; i != len; i++)
			for (int j = 0, len2 = matrix[i].length; j != len2; j++) {
				double real = random.nextDouble();
				matrix[i][j] = random.nextDouble() > 0.5 ? real : -real;
			}
	}
}
