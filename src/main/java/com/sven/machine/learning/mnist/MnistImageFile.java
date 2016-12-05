package com.sven.machine.learning.mnist;

import java.io.IOException;

public class MnistImageFile extends MnistFile
{

    private int rows;
    private int cols;

    public MnistImageFile(String name, String mode) throws IOException
    {
        super(name, mode);

        // read header information
        rows = readInt();
        cols = readInt();
    }

    public double[][] readImage() throws IOException
    {

        double[][] data = new double[rows][cols];
        for (int row = 0; row < rows; row++)
        {
            for (int col = 0; col < cols; col++)
            { 
                int b  =readUnsignedByte();
                if (b != 0) {
                    data[row][col] =  (double)Math.round(b / 255f * 100) / 100;
                }
                
            }
        }
        return data;
    }

    public byte[][] readImagesUnsafe(int nImages) throws IOException
    {
        byte[][] out = new byte[nImages][0];
        for (int i = 0; i < nImages; i++)
        {
            out[i] = new byte[rows * cols];
            read(out[i]);
        }
        return out;
    }

    @Override
    protected int getMagicNumber()
    {
        return 2051;
    }

    @Override
    public int getEntryLength()
    {
        return cols * rows;
    }

    @Override
    public int getHeaderSize()
    {
        return 8 + 4 + 4; // magic number, rows and columns
    }
}
