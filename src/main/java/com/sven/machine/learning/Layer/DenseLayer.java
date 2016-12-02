package com.sven.machine.learning.Layer;

import com.sven.machine.learning.enums.ActivationType;

public class DenseLayer extends Layer
{

    private int outputNumber;
    private ActivationType activationType;
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
    
    
}
