package com.sven.machine.learning.network;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.List;

import org.apache.commons.io.FileUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.gson.Gson;
import com.sven.machine.learning.enums.LayerType;
import com.sven.machine.learning.layer.Layer;
import com.sven.machine.learning.mnist.MnistData;
import com.sven.machine.learning.utils.MathUtil;

public class CNN
{
    Logger log = LoggerFactory.getLogger(this.getClass());
    static String projectPath = System.getProperty("user.dir");

    private CNNConfig config;
    private String configFilePath = projectPath + "\\src\\main\\resources\\model\\";

    public CNN(CNNConfig config)
    {
        this.config = config;
    }

    public void save(String fileName) throws IOException
    {
        List<Layer> layers = this.config.getLayers();
        int index = 0;
        for (Layer layer : layers)
        {

            Gson gson = new Gson();
            String json = gson.toJson(layer);
            FileUtils.writeStringToFile(
                    new File(configFilePath + fileName + "_" + index),
                    json,
                    Charset.defaultCharset(),
                    false);
            index++;
        }

    }

    public void load(String fileName) throws IOException
    {
        List<Layer> layers = config.getLayers();
        int index = 0;
        for (Layer layer : layers)
        {
            String json = FileUtils.readFileToString(
                    new File(configFilePath + fileName + "_" + index),
                    Charset.defaultCharset());
            layers.set(index, new Gson().fromJson(json, layer.getClass()));
            index++;
        }
        this.updateRelations();
    }

    public void init()
    {
        updateRelations();
        for (Layer layer : config.getLayers())
        {

            layer.init();
        }
        
    }

    protected void updateRelations()
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
            forward(data);
            
            if (isCorrect(data))
            {
                correctNumber++;
            }
            ;
        }

        double rate = Math.round(correctNumber * 10000d / batchSize) / 100d;
        log.info(
                "correct/batch[" + correctNumber + "/" + batchSize + "]" + "=" + rate
                        + "%");
    }

    public void train(List<MnistData> batch)
    {
        int batchSize = batch.size();
        int correctNumber = 0;
        for (MnistData data : batch)
        {
            if (train(data))
            {
                correctNumber++;
            }
            ;
        }

        double rate = Math.round(correctNumber * 100 / batchSize);
        log.info(
                "correct/batch[" + correctNumber + "/" + batchSize + "]" + "=" + rate
                        + "%");
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
        // look for result
        double[][][] maps = this.config.getLayers().stream().filter(
                s -> s.getLayerType() == LayerType.outputLayer).findFirst().get().getMaps();
        double[] temp = new double[maps.length];
        for (int i = 0; i < temp.length; i++)
        {
            temp[i] = maps[i][0][0];
        }
        
        log.debug("test with result:" + MathUtil.getMaxIndex(temp));
        
        return (data.getLabel() == MathUtil.getMaxIndex(temp));
    }

    protected void forward(MnistData record)
    {
        for (Layer layer : config.getLayers())
        {
            layer.forward(record);
        }
    }

    protected void bp(MnistData record)
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

    public CNNConfig getConfig()
    {
        return config;
    }

    public void setConfig(CNNConfig config)
    {
        this.config = config;
    }

}
