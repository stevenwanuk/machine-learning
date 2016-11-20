package com.sven.machine.learning.mnist;

import java.io.IOException;

import org.springframework.util.StringUtils;

public class MnistDataSet {

	MnistLabelFile labelFile;
	MnistImageFile imageFile;

	public MnistDataSet(String labelFileName, String imageFileName) {

		try {
			if (!StringUtils.isEmpty(labelFileName)) {
				labelFile = new MnistLabelFile(labelFileName, "r");
			}
			if (!StringUtils.isEmpty(imageFileName)) {
				imageFile = new MnistImageFile(imageFileName, "r");
			}

		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	public void setCurrent(int index) {

		try {
			if (labelFile != null) {
				labelFile.setCurrentIndex(index);
			}
			if (imageFile != null) {

				imageFile.setCurrentIndex(index);
			}
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}

	public boolean hasNext() {
		boolean hasLabel = true;
		boolean hasImage = false;
		try {
			if (labelFile != null && labelFile.getCurrentIndex() > labelFile.getCount()) {

				hasLabel = false;
			}

			if (imageFile != null && imageFile.getCurrentIndex() < imageFile.getCount()) {

				hasImage = true;
			}

		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return hasLabel && hasImage;
	}

	public MnistData read() {

		MnistData data = new MnistData();
		if (labelFile != null) {
			try {
				data.setLabel(labelFile.readLabel());
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}

		if (imageFile != null) {
			try {
				data.setImageByte(imageFile.readImage());
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		return data;
	}

	public void close() {

		if (labelFile != null) {
			try {
				labelFile.close();
			} catch (IOException e) {
			}
			labelFile = null;
		}
		if (imageFile != null) {
			try {
				imageFile.close();
			} catch (IOException e) {
			}
			imageFile = null;
		}

	}

	public static void main(String[] args) {

		MnistDataSet dataSet = new MnistDataSet(
				"C:\\workspace\\machineLearning\\machine-learning-example\\src\\main\\resources\\mnist\\train-labels.idx1-ubyte",
				"C:\\workspace\\machineLearning\\machine-learning-example\\src\\main\\resources\\mnist\\train-images.idx3-ubyte");
		int i = 0;
		while (dataSet.hasNext()) {
			System.out.println(i++ +" " + dataSet.read());
		}
	}
}
