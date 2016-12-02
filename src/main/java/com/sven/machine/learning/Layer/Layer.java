package com.sven.machine.learning.Layer;

import com.sven.machine.learning.enums.LayerType;

public abstract class Layer
{

    private int layerIndex;
    private LayerType layerType;

    public int getLayerIndex()
    {
        return layerIndex;
    }

    public void setLayerIndex(int layerIndex)
    {
        this.layerIndex = layerIndex;
    }

    public LayerType getLayerType()
    {
        return layerType;
    }

    public void setLayerType(LayerType layerType)
    {
        this.layerType = layerType;
    }

}
