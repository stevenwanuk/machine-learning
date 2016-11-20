package com.sven.machine.learning.mnist;

import java.io.IOException;
import java.io.RandomAccessFile;

/**
 * http://yann.lecun.com/exdb/mnist/
 * @author KWan
 *
 */
public abstract class MnistFile extends RandomAccessFile {

	private int count;

	protected abstract int getMagicNumber();

	protected abstract int getEntryLength();

	protected abstract int getHeaderSize();
	
	public MnistFile(String name, String mode) throws IOException {
		super(name, mode);
		// TODO Auto-generated constructor stub
		int magicNumber = getMagicNumber();
		if (magicNumber != readInt()) {
			throw new RuntimeException("fail to read file without correct magic number:" + magicNumber + "");
		}
		count = readInt();
	}

	

	public long getCurrentIndex() throws IOException {
		return (getFilePointer() - getHeaderSize()) / getEntryLength() + 1;
	}

	public void setCurrentIndex(long curr) throws IOException {
		if (curr < 0 || curr > count) {
			throw new IndexOutOfBoundsException(curr + " is not in the range 0 to " + count);
		}
		seek(getHeaderSize() + curr * getEntryLength());
	}

	public void next() throws IOException {
		if (getCurrentIndex() < count) {
			skipBytes(getEntryLength());
		}
	}

	public void prev() throws IOException {
		if (getCurrentIndex() > 0) {
			seek(getFilePointer() - getEntryLength());
		}
	}

	public int getCount() {
		return count;
	}
	
    public static void main(String[] args) 
    {
    
    	try (MnistImageFile file = new MnistImageFile("C:\\workspace\\machineLearning\\machine-learning-example\\src\\main\\resources\\mnist\\train-images.idx3-ubyte", "r")){
			for(int i = 0 ; i < file.getCount(); i++) 
			{
			
				System.out.println(i + "->" + file.readImage());
			}
			
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
    }
}
