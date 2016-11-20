package com.sven.machine.learning.mnist;

import java.io.IOException;

public class MnistLabelFile extends MnistFile{
	
    public MnistLabelFile(String name, String mode) throws  IOException {
        super(name, mode);
    }
    
    public int readLabel() throws IOException {
        return readUnsignedByte();
    }

    public int[] readLabels(int num) throws IOException {
        int[] out = new int[num];
        for( int i=0; i<num; i++ ) out[i] = readLabel();
        return out;
    }

	
    @Override
    protected int getMagicNumber() {
        return 2049;
    }
    
    @Override
    public int getEntryLength() {
        return 1;
    }
    
    @Override
    public int getHeaderSize() {
        return 8;//magic number
    }
    
    public static void main(String[] args) 
    {
    
    	try (MnistLabelFile file = new MnistLabelFile("C:\\workspace\\machineLearning\\machine-learning-example\\src\\main\\resources\\mnist\\train-labels.idx1-ubyte", "r")){
			for(int i = 0 ; i < file.getCount(); i++) 
			{
			
				System.out.println(i + "->" + file.read());
			}
			
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
    }
}
