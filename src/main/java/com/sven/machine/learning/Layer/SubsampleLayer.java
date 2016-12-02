package com.sven.machine.learning.Layer;

import com.sven.machine.learning.enums.PoolingType;
import com.sven.machine.model.Matrix;

public class SubsampleLayer extends Layer
{
    public PoolingType poolingType;
    private Matrix<Integer> kernelSize;
    private Matrix<Integer> stride;
    private Matrix<Integer> padding;
    public PoolingType getPoolingType()
    {
        return poolingType;
    }
    public void setPoolingType(PoolingType poolingType)
    {
        this.poolingType = poolingType;
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
