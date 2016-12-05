package com.sven.machine.learning.network;

import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.sven.machine.learning.enums.LayerType;
import com.sven.machine.learning.layer.Layer;
import com.sven.machine.learning.mnist.MnistData;
import com.sven.machine.learning.utils.MathUtil;

public class CNN
{
    Logger log = LoggerFactory.getLogger(this.getClass());

    private CNNConfig config;

    public CNN(CNNConfig config)
    {
        this.config = config;
    }

    public void init()
    {

        Layer prevLayer = null;
        int index = 0;
        for (Layer layer : config.getLayers())
        {

            layer.setLayerIndex(index);
            if (prevLayer != null)
            {
                prevLayer.setAfterLayer(layer);
                layer.setPrevLayer(prevLayer);
            }
            layer.init();
            log.debug(
                    "init layer:" + index + " " + layer.getLayerType() + " "
                            + layer.getMapSize() + " "
                            + layer.getMapNumber());

            prevLayer = layer;
            index++;
        }
    }
    
    public void test(List<MnistData> batch) 
    {
        int batchSize = batch.size();
        int correctNumber = 0;
        for (MnistData data : batch)
        {
            if (train(data)) {
                correctNumber++;
            };
        }
        
        double rate = Math.round(correctNumber * 100 / batchSize);
        log.info("correct/batch[" + correctNumber+"/" + batchSize + "]" + "="  +  rate + "%");
    }

    public void train(List<MnistData> batch)
    {
        int batchSize = batch.size();
        int correctNumber = 0;
        for (MnistData data : batch)
        {
            if (train(data)) {
                correctNumber++;
            };
        }
        
        
        double rate = Math.round(correctNumber * 100 / batchSize);
        log.info("correct/batch[" + correctNumber+"/" + batchSize + "]" + "="  +  rate + "%");
    }

    public boolean train(MnistData data)
    {
        forward(data);
        
        bp(data);
        learning();
        
        return isCorrect(data);
    }
    
    protected boolean isCorrect(MnistData data) 
    {
        //look for result
        double[][][] maps = this.config.getLayers().stream().filter(s -> s.getLayerType() == LayerType.outputLayer).findFirst().get().getMaps();
        double[] temp = new double[maps.length];
        for (int i = 0; i < temp.length; i++)
        {
            temp[i] = maps[i][0][0];
        }
        return (data.getLabel() == MathUtil.getMaxIndex(temp));        
    }
    

    private void forward(MnistData record)
    {
        for (Layer layer : config.getLayers())
        {
            layer.forward(record);
        }
    }

    private void bp(MnistData record)
    {
        for (int i = 0; i < config.getLayers().size(); i++)
        {

            config.getLayers().get(config.getLayers().size() - 1 - i).bp(record);
        }
    }

    private void learning()
    {
        for (Layer layer : config.getLayers())
        {
            layer.learnFromErrors();
        }
    }

    protected boolean backPropagation(MnistData record)
    {
        // boolean result = setOutLayerErrors(record);
        // setHiddenLayerErrors();
        // return result;
        return false;
    }

    public void verify(MnistData data)
    {

    }

    public CNNConfig getConfig()
    {
        return config;
    }

    public void setConfig(CNNConfig config)
    {
        this.config = config;
    }

}
