package com.sven.machine.learning.utils;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class LogUtil
{
    static Logger log = LoggerFactory.getLogger(LogUtil.class);

    public static void logMaps(double[][][] maps)
    {

        for (int index = 0; index < maps.length; index++)
        {
            log.debug("map[" + index + "]");

            String str = "";
            for (int i = 0; i < maps[0].length; i++)
            {

                for (int j = 0; j < maps[0][0].length; j++)
                {
                    str += "{[" + i + "," + j + "]=" + maps[index][i][j] + "}";
                }
            }
            log.debug("map[" + index + "]=" + str);
        }
    }
}
