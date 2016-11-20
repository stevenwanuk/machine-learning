package com.sven.machine.learning.mnist;

public class MnistData {

	private int label;
	private int[][] imageByte;
	public int getLabel() {
		return label;
	}
	public void setLabel(int label) {
		this.label = label;
	}
	public int[][] getImageByte() {
		return imageByte;
	}
	public void setImageByte(int[][] imageByte) {
		this.imageByte = imageByte;
	}
	@Override
	public String toString() {
		
		return "label:" + label + "," + "imageBytes:" + imageByte;
	}
	
}
