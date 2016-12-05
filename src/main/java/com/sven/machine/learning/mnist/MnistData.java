package com.sven.machine.learning.mnist;

public class MnistData {

	private int label;
	private double[][] imageByte;
	public int getLabel() {
		return label;
	}
	public void setLabel(int label) {
		this.label = label;
	}
	public double[][] getImageByte() {
		return imageByte;
	}
	public void setImageByte(double[][] imageByte) {
		this.imageByte = imageByte;
	}
	@Override
	public String toString() {
		
		return "label:" + label + "," + "imageBytes:" + imageByte;
	}
	
}
