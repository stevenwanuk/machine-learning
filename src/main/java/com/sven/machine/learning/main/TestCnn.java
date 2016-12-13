package com.sven.machine.learning.main;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.sven.machine.learning.mnist.MnistData;
import com.sven.machine.learning.utils.ImageUtil;
import com.sven.machine.learning.utils.MathUtil;

public class TestCnn
{
    
    static Logger log = LoggerFactory.getLogger(TestCnn.class);
    
    public static void main(String[] args) throws IOException 
    {
    
        String[] images = new String[] {"1.JPG","2.JPG","3.JPG","5.JPG"};
        int[] labels = new int[]{1,2,3,5}; 
        log.info("********start to test**********");
        MnistCNN.load().testWithLog(buildBatch(images, labels));
        
    }
    public static List<MnistData> buildBatch(String[] imageNames, int[] labels) throws IOException 
    {
    
        List<MnistData> batch = new ArrayList<>();
        int index=0;
        for(String imageFileName : imageNames) {
            
            double[][] d = ImageUtil.getImage(imageFileName);
            MnistData md = new MnistData();
            md.setImageByte(d);
            md.setLabel(labels[index++]);
            MathUtil.normalizeForSigmoid(md);
            batch.add(md);
        }
        return batch;
    }
}
