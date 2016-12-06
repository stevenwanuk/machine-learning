package com.sven.machine.learning.utils;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.Toolkit;
import java.awt.image.BufferedImage;
import java.awt.image.FilteredImageSource;
import java.awt.image.ImageFilter;
import java.awt.image.ImageProducer;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;
import javax.swing.GrayFilter;

public class ImageUtil
{
    static String projectPath = System.getProperty("user.dir");

    public static double[][] getImage(String fileName) throws IOException {
        BufferedImage bufferedImage = ImageIO.read(
                new File(projectPath + "\\src\\main\\resources\\sample\\hw\\" + fileName));

        ImageFilter filter = new GrayFilter(true, 20);
        ImageProducer producer =
                new FilteredImageSource(bufferedImage.getSource(), filter);
        Image mage = Toolkit.getDefaultToolkit().createImage(producer);
        
        Image scaledImage1 =
                mage.getScaledInstance(28, 28, Image.SCALE_AREA_AVERAGING);
        saveTookitImage(scaledImage1, scaledImage1.getWidth(null), scaledImage1.getHeight(null), "3.2");

        BufferedImage bimage1 = new BufferedImage(scaledImage1.getWidth(null), scaledImage1.getHeight(null), BufferedImage.TYPE_INT_ARGB);
        bimage1.getGraphics().drawImage(scaledImage1, 0, 0, null);
        
        int rc = 0;
        int rg = 0;
        int rb = 0;
        for (int i = 0; i < 28; i++)
        {
            for (int j = 0; j < 28; j++)
            {
                int rgb = bimage1.getRGB(i, j);
                rc += (rgb & 0xff0000) >> 16;
                rg += (rgb & 0xff00) >> 8;
                rb += (rgb & 0xff);
            }
        }

        BufferedImage image = new BufferedImage(112, 112, BufferedImage.TYPE_INT_RGB);
        Graphics2D graphics = image.createGraphics();
        graphics.setPaint(new Color(255, 255, 255));
        graphics.fillRect(0, 0, image.getWidth(), image.getHeight());

        int ac = rc / (28 * 28);
        int ag = rg / (28 * 28);
        int ab = rb / (28 * 28);
        double[][] d = new double[28][28];
        for (int i = 0; i < 28; i++)
        {
            for (int j = 0; j < 28; j++)
            {
                int rgb = bimage1.getRGB(i, j);
                int r = (rgb & 0xff0000) >> 16;
                int g = (rgb & 0xff00) >> 8;
                int b = (rgb & 0xff);

                if (Math.abs(r - ac) > 5 || Math.abs(g - ag) > 5
                        || Math.abs(b - ab) > 5)
                {
                    image.setRGB(i, j, 1, 1, new int[] { 0, 0, 0 }, 0, 1);
                    d[i][j] = 255;
                }

            }
        }
        
        ImageIO.write(
                image,
                "png",
                new File(projectPath + "\\src\\main\\resources\\sample\\hw\\" + "2.6"
                        + ".png"));
        
        
        
//        Image scaledImage2 = image.getScaledInstance(28, 28, Image.SCALE_SMOOTH);
//        saveTookitImage(scaledImage2, scaledImage2.getWidth(null), scaledImage2.getHeight(null), "9");
//        scaledImage2.flush();
//        
//        BufferedImage image2 = new BufferedImage(28, 28, BufferedImage.TYPE_INT_RGB);
//        image2.getGraphics().drawImage(scaledImage2, 0, 0, null);
//        saveTookitImage(image2, image2.getWidth(null), image2.getHeight(null), "3.3");
//        
//        
//        double[][] d = new double[image2.getWidth()][image2.getHeight()];
//        for(int i = 0; i< image2.getWidth(); i++) 
//        {
//        
//            for(int j = 0; j < image2.getHeight(); j++) {
//                
//                
//                
//                int rgb = image2.getRGB(i, j);
//                int gray= rgb& 0xFF;
//                if (gray <= 0) {
//                    d[i][j] = 0;
//                } else {
//                    d[i][j] = 100;    
//                }
//                d[i][j] = rgb;
//                
//            }
//        }
        
        return d;
    }

    public static void saveTookitImage(Image image, int width, int height, String fileName)
            throws IOException
    {
        BufferedImage bi = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
        bi.getGraphics().drawImage(image, 0, 0, null);
        ImageIO.write(
                bi,
                "png",
                new File(projectPath + "\\src\\main\\resources\\sample\\hw\\" + fileName
                        + ".png"));
    }
}
