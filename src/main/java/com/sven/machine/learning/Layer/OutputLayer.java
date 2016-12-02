package com.sven.machine.learning.Layer;

import com.sven.machine.learning.enums.ActivationType;
import com.sven.machine.learning.enums.LostFunctionType;

public class OutputLayer extends Layer
{

    private int outputNumber;
    private ActivationType activationType;
    private LostFunctionType lostFunctionType;
    public int getOutputNumber()
    {
        return outputNumber;
    }
    public void setOutputNumber(int outputNumber)
    {
        this.outputNumber = outputNumber;
    }
    public ActivationType getActivationType()
    {
        return activationType;
    }
    public void setActivationType(ActivationType activationType)
    {
        this.activationType = activationType;
    }
    public LostFunctionType getLostFunctionType()
    {
        return lostFunctionType;
    }
    public void setLostFunctionType(LostFunctionType lostFunctionType)
    {
        this.lostFunctionType = lostFunctionType;
    }
    
}
