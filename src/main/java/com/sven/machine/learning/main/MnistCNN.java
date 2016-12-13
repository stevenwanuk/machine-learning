package com.sven.machine.learning.main;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import javax.imageio.ImageIO;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.sven.machine.learning.enums.PoolingType;
import com.sven.machine.learning.layer.ConvolutionalLayer;
import com.sven.machine.learning.layer.InputLayer;
import com.sven.machine.learning.layer.Layer;
import com.sven.machine.learning.layer.OutputLayer;
import com.sven.machine.learning.layer.SubsamplingLayer;
import com.sven.machine.learning.mnist.MnistData;
import com.sven.machine.learning.mnist.MnistDataSet;
import com.sven.machine.learning.model.Matrix;
import com.sven.machine.learning.network.CNN;
import com.sven.machine.learning.network.CNNConfig;
import com.sven.machine.learning.utils.ImageUtil;

public class MnistCNN
{
    static Logger log = LoggerFactory.getLogger(MnistCNN.class);
    static String projectPath = System.getProperty("user.dir");
    public static MnistDataSet readTrainData()
    {
       

        MnistDataSet dataSet = new MnistDataSet(
                projectPath + "\\src\\main\\resources\\mnist\\train-labels.idx1-ubyte",
                projectPath + "\\src\\main\\resources\\mnist\\train-images.idx3-ubyte");
        return dataSet;
    }

    public static MnistDataSet readTestData()
    {
        MnistDataSet dataSet = new MnistDataSet(
                projectPath + "\\src\\main\\resources\\mnist\\t10k-labels.idx1-ubyte",
                projectPath + "\\src\\main\\resources\\mnist\\t10k-images.idx3-ubyte");
        return dataSet;
    }

    public static CNNConfig buildNetwork()
    {
        CNNConfig config = new CNNConfig();
        config.setLearningRate(100);
        config.setLayers(buildLayers());
        return config;
    }

    public static List<Layer> buildLayers()
    {

        List<Layer> layers = new ArrayList<>();

        InputLayer i0 = new InputLayer();
        i0.setMapSize(new Matrix<Integer>(28, 28));
        layers.add(i0);

        // 28*28
        ConvolutionalLayer c0 = new ConvolutionalLayer();
        c0.setKernelSize(new Matrix<>(5, 5));
        c0.setPadding(new Matrix<>(0, 0));
        c0.setStride(new Matrix<>(1, 1));
        c0.setMapNumber(6);
        c0.setChannelSize(1);
        layers.add(c0);
        // 24*24*6

        SubsamplingLayer s0 = new SubsamplingLayer();
        s0.setKernelSize(new Matrix<>(2, 2));
        s0.setStride(new Matrix<>(2, 2));
        s0.setPadding(new Matrix<>(0, 0));
        s0.setPoolingType(PoolingType.max);
        layers.add(s0);
        // 12*12*6
        ConvolutionalLayer c1 = new ConvolutionalLayer();
        c1.setKernelSize(new Matrix<>(5, 5));
        c1.setPadding(new Matrix<>(0, 0));
        c1.setStride(new Matrix<>(1, 1));
        c1.setMapNumber(12);
        layers.add(c1);
        // 8*8*12
        SubsamplingLayer s1 = new SubsamplingLayer();
        s1.setKernelSize(new Matrix<>(2, 2));
        s1.setStride(new Matrix<>(2, 2));
        s1.setPadding(new Matrix<>(0, 0));
        s1.setPoolingType(PoolingType.max);
        layers.add(s1);
        // 12
        OutputLayer o0 = new OutputLayer();
        o0.setOutputNumber(10);
        layers.add(o0);

        return layers;
    }

    public static void main(String[] args) throws IOException
    {
         CNN cnn = new CNN(buildNetwork());
         cnn.init();
         train(cnn);
         cnn.load("config");
         test(cnn);
    }
    
    public static void testhw(CNN cnn) throws IOException{
        
        double[][] d = ImageUtil.getImage("5.jpg");
        
        List<MnistData> batch = new ArrayList<>();
        MnistData md = new MnistData();
        md.setImageByte(d);
        md.setLabel(5);
        saveToImage(d, "test");
        normalizeForSigmoid(md);
        batch.add(md);
        cnn.test(batch);
    }

    public static void saveToImage(double[][] imageD, String fileName) throws IOException
    {
        BufferedImage image = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D graphics = image.createGraphics();

        graphics.setPaint(new Color(255, 255, 255));
        graphics.fillRect(0, 0, image.getWidth(), image.getHeight());
        for (int i = 0; i < imageD.length; i++)
        {
            for (int j = 0; j < imageD[0].length; j++)
            {
                int gray = (int) imageD[i][j];
                if (gray > 30)

                    image.setRGB(i, j, 1, 1, new int[] { 0, 0, 0 }, 0, 1);

            }
        }
        ImageIO.write(
                image,
                "png",
                new File(projectPath +"\\src\\main\\resources\\sample\\" + fileName + ".png"));
    }

    public static void train(CNN cnn) throws IOException
    {
        log.info("********start to train**********");
        int batchSize = 100;
        int epoch = 1000;

        MnistDataSet dataSet = readTrainData();
        for (int i = 0; i < epoch; i++)
        {
            int batchIndex = 0;
            while (dataSet.hasNext())
            {
                log.info("start training, epoch:" + i + " batchIndex:" + batchIndex++);
                List<MnistData> batch = new ArrayList<>();
                for (int j = 0; j < batchSize && dataSet.hasNext(); j++)
                {
                    MnistData data = dataSet.read();
                    normalizeForSigmoid(data);
                    batch.add(data);
                }
                cnn.train(batch);
                cnn.save("config");

            }
            dataSet.setCurrent(0);
        }
    }

    public static void test(CNN cnn)
    {
        log.info("********start to test**********");
        MnistDataSet dataSet = readTestData();

        List<MnistData> batch = new ArrayList<>();
        while (dataSet.hasNext())
        {
            MnistData data = dataSet.read();
            normalizeForSigmoid(data);
            batch.add(data);
        }
        cnn.test(batch);
    }

    private static void normalizeForSigmoid(MnistData data)
    {

        double[][] d = data.getImageByte();
        for (int i = 0; i < d.length; i++)
        {
            for (int j = 0; j < d[0].length; j++)
            {

                // d[i][j] = (double) Math.round(d[i][j] / 255f * 100) / 100;
                d[i][j] = d[i][j] > 30 ? 1 : 0;
            }

        }
    }
    
    private static void normalizeForRelu(MnistData data)
    {

        double[][] d = data.getImageByte();
        for (int i = 0; i < d.length; i++)
        {
            for (int j = 0; j < d[0].length; j++)
            {

               
                d[i][j] = d[i][j] > 30 ? 1 : -1;
            }

        }
    }
}
