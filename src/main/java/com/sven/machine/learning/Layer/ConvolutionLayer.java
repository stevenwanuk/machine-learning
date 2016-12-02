package com.sven.machine.learning.Layer;

import com.sven.machine.model.Matrix;

public class ConvolutionLayer extends Layer
{

    private int channelSize;
    private int filterSize;
    private Matrix<Integer> kernelSize;
    private Matrix<Integer> stride;
    private Matrix<Integer> padding;

    public int getChannelSize()
    {
        return channelSize;
    }

    public void setChannelSize(int channelSize)
    {
        this.channelSize = channelSize;
    }

    public int getFilterSize()
    {
        return filterSize;
    }

    public void setFilterSize(int filterSize)
    {
        this.filterSize = filterSize;
    }

    public Matrix<Integer> getKernelSize()
    {
        return kernelSize;
    }

    public void setKernelSize(Matrix<Integer> kernelSize)
    {
        this.kernelSize = kernelSize;
    }

    public Matrix<Integer> getStride()
    {
        return stride;
    }

    public void setStride(Matrix<Integer> stride)
    {
        this.stride = stride;
    }

    public Matrix<Integer> getPadding()
    {
        return padding;
    }

    public void setPadding(Matrix<Integer> padding)
    {
        this.padding = padding;
    }

}
