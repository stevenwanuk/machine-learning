package com.sven.machinelearning;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class BpDeep {
	public double[][] layer;// 神经网络各层节点
	public double[][] layerErr;// 神经网络各节点误差
	public double[][][] layer_weight;// 各层节点权重
	public double[][][] layer_weight_delta;// 各层节点权重动量
	public double mobp;// 动量系数
	public double rate;// 学习系数

	public BpDeep(int[] layernum, double rate, double mobp) {
		this.mobp = mobp;
		this.rate = rate;
		layer = new double[layernum.length][];
		layerErr = new double[layernum.length][];
		layer_weight = new double[layernum.length][][];
		layer_weight_delta = new double[layernum.length][][];
		Random random = new Random();
		for (int l = 0; l < layernum.length; l++) {
			layer[l] = new double[layernum[l]];
			layerErr[l] = new double[layernum[l]];
			if (l + 1 < layernum.length) {
				layer_weight[l] = new double[layernum[l] + 1][layernum[l + 1]];
				layer_weight_delta[l] = new double[layernum[l] + 1][layernum[l + 1]];
				for (int j = 0; j < layernum[l] + 1; j++)
					for (int i = 0; i < layernum[l + 1]; i++)
						layer_weight[l][j][i] = random.nextDouble();// 随机初始化权重
			}
		}
	}

	// 逐层向前计算输出
	public double[] computeOut(double[] in) {
		for (int l = 1; l < layer.length; l++) {
			for (int j = 0; j < layer[l].length; j++) {
				double z = layer_weight[l - 1][layer[l - 1].length][j];
				for (int i = 0; i < layer[l - 1].length; i++) {
					layer[l - 1][i] = l == 1 ? in[i] : layer[l - 1][i];
					z += layer_weight[l - 1][i][j] * layer[l - 1][i];
				}
				layer[l][j] = 1 / (1 + Math.exp(-z));
			}
		}
		return layer[layer.length - 1];
	}

	// 逐层反向计算误差并修改权重
	public void updateWeight(double[] tar) {
		int l = layer.length - 1;
		for (int j = 0; j < layerErr[l].length; j++)
			layerErr[l][j] = layer[l][j] * (1 - layer[l][j]) * (tar[j] - layer[l][j]);

		while (l-- > 0) {
			for (int j = 0; j < layerErr[l].length; j++) {
				double z = 0.0;
				for (int i = 0; i < layerErr[l + 1].length; i++) {
					z = z + l > 0 ? layerErr[l + 1][i] * layer_weight[l][j][i] : 0;
					layer_weight_delta[l][j][i] = mobp * layer_weight_delta[l][j][i]
							+ rate * layerErr[l + 1][i] * layer[l][j];// 隐含层动量调整
					layer_weight[l][j][i] += layer_weight_delta[l][j][i];// 隐含层权重调整
					if (j == layerErr[l].length - 1) {
						layer_weight_delta[l][j + 1][i] = mobp * layer_weight_delta[l][j + 1][i]
								+ rate * layerErr[l + 1][i];// 截距动量调整
						layer_weight[l][j + 1][i] += layer_weight_delta[l][j + 1][i];// 截距权重调整
					}
				}
				layerErr[l][j] = z * layer[l][j] * (1 - layer[l][j]);// 记录误差
			}
		}
	}

	public void train(double[] in, double[] tar) {
		double[] out = computeOut(in);
		updateWeight(tar);
	}

	public static void main2(String[] args) {
		// 初始化神经网络的基本配置
		// 第一个参数是一个整型数组，表示神经网络的层数和每层节点数，比如{3,10,10,10,10,2}表示输入层是3个节点，输出层是2个节点，中间有4层隐含层，每层10个节点
		// 第二个参数是学习步长，第三个参数是动量系数
		BpDeep bp = new BpDeep(new int[] { 2, 10, 2 }, 0.15, 0.8);

		// 设置样本数据，对应上面的4个二维坐标数据
		double[][] data = new double[][] { { 1, 2 }, { 2, 2 }, { 1, 1 }, { 2, 1 } };
		// 设置目标数据，对应4个坐标数据的分类
		double[][] target = new double[][] { { 1, 0 }, { 0, 1 }, { 0, 1 }, { 1, 0 } };

		// 迭代训练5000次
		for (int n = 0; n < 5000; n++)
			for (int i = 0; i < data.length; i++)
				bp.train(data[i], target[i]);

		// 根据训练结果来检验样本数据
		for (int j = 0; j < data.length; j++) {
			double[] result = bp.computeOut(data[j]);
			System.out.println(Arrays.toString(data[j]) + ":" + Arrays.toString(result));
		}

		// 根据训练结果来预测一条新数据的分类
		double[] x = new double[] { 3, 1 };
		double[] result = bp.computeOut(x);
		System.out.println(Arrays.toString(x) + ":" + Arrays.toString(result));
	}

	public static void main3(String[] args) throws IOException {
		BpDeep bp = new BpDeep(new int[] { 32, 10, 4 }, 0.15, 0.8);

		Random random = new Random();
		List<Integer> list = new ArrayList<Integer>();
		for (int i = 0; i != 1000; i++) {
			int value = random.nextInt();
			list.add(value);
		}

		for (int i = 0; i != 200; i++) {
			for (int value : list) {
				double[] real = new double[4];
				if (value >= 0)
					if ((value & 1) == 1)
						real[0] = 1;
					else
						real[1] = 1;
				else if ((value & 1) == 1)
					real[2] = 1;
				else
					real[3] = 1;
				double[] binary = new double[32];
				int index = 31;
				do {
					binary[index--] = (value & 1);
					value >>>= 1;
				} while (value != 0);

				bp.train(binary, real);
			}
		}

		System.out.println("训练完毕，下面请输入一个任意数字，神经网络将自动判断它是正数还是复数，奇数还是偶数。");

		while (true) {
			byte[] input = new byte[10];
			System.in.read(input);
			Integer value = Integer.parseInt(new String(input).trim());
			int rawVal = value;
			double[] binary = new double[32];
			int index = 31;
			do {
				binary[index--] = (value & 1);
				value >>>= 1;
			} while (value != 0);

			double[] result = bp.computeOut(binary);

			double max = -Integer.MIN_VALUE;
			int idx = -1;

			for (int i = 0; i != result.length; i++) {
				if (result[i] > max) {
					max = result[i];
					idx = i;
				}
			}

			switch (idx) {
			case 0:
				System.out.format("%d是一个正奇数\n", rawVal);
				break;
			case 1:
				System.out.format("%d是一个正偶数\n", rawVal);
				break;
			case 2:
				System.out.format("%d是一个负奇数\n", rawVal);
				break;
			case 3:
				System.out.format("%d是一个负偶数\n", rawVal);
				break;
			}
		}
	}

	public static void main(String[] args) throws IOException {
		BpDeep bp = new BpDeep(new int[] { 1, 32, 32, 32, 4 }, 0.15, 0.8);

		Random random = new Random();
		List<Integer> list = new ArrayList<Integer>();
		for (int i = -10000; i < 10000; i++) {
			int value = i;
			list.add(value);
		}

		for (int i = 0; i < 10; i++) {
			for (int value : list) {
				System.out.println(value);
				double[] target;
				if (value > 0) {

					if (value % 2 == 1) {

						target = new double[] { 1, 0, 0, 0 };
					} else {
						target = new double[] { 0, 1, 0, 0 };
					}
				} else {

					if (value % 2 == 1) {

						target = new double[] { 0, 0, 1, 0 };
					} else {
						target = new double[] { 0, 0, 0, 1 };
					}
				}

				//bp.train(new double[] { Integer. }, target);
			}
		}

		System.out.println("训练完毕，下面请输入一个任意数字，神经网络将自动判断它是正数还是复数，奇数还是偶数。");

		while (true) {
			byte[] input = new byte[10];
			System.in.read(input);
			Integer value = Integer.parseInt(new String(input).trim());

			double[] result = bp.computeOut(new double[] { value / 10000 });
			System.out.println(result[0]);
			System.out.println(result[1]);
			System.out.println(result[2]);
			System.out.println(result[3]);

		}
	}

}
