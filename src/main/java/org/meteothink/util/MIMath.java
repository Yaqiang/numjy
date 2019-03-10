/* Copyright 2012 Yaqiang Wang,
 * yaqiang.wang@gmail.com
 * 
 * This library is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 * 
 * This library is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser
 * General Public License for more details.
 */
package org.meteothink.util;

import java.awt.Point;
import java.awt.Rectangle;
import java.util.ArrayList;
import java.util.List;

/**
 * MeteoInfo Math class
 *
 * @author Yaqiang Wang
 */
public class MIMath {

    /**
     * Determine if two double data equal
     *
     * @param a double a
     * @param b double b
     * @return boolean
     */
    public static boolean doubleEquals(double a, double b) {
        double difference = Math.abs(a * 0.00001);
        if (Math.abs(a - b) <= difference) {
            return true;
        } else {
            return false;
        }
    }

    /**
     * Determine if two double data equal
     *
     * @param a double a
     * @param b double b
     * @return boolean
     */
    public static boolean doubleEquals_Abs(double a, double b) {
        if (Math.abs(a - b) < 0.0000001) {
            return true;
        } else {
            return false;
        }
    }

    /**
     * Get mininum and maximum values
     *
     * @param S Data array
     * @param unDef Undefined data
     * @return Minimum and Maximum data array
     */
    public static double[] getMinMaxValue(double[] S, double unDef) {
        int i, validNum;
        double min = unDef, max = unDef;

        validNum = 0;
        for (i = 0; i < S.length; i++) {
            if (!(doubleEquals(S[i], unDef))) {
                validNum++;
                if (validNum == 1) {
                    min = S[i];
                    max = min;
                } else {
                    if (S[i] < min) {
                        min = S[i];
                    }
                    if (S[i] > max) {
                        max = S[i];
                    }
                }
            }
        }

        return new double[]{min, max};
    }

    /**
     * Get mininum and maximum values
     *
     * @param S Data list
     * @param unDef Undefined data
     * @return Minimum and Maximum data array
     */
    public static double[] getMinMaxValue(List<Double> S, double unDef) {
        int i, validNum;
        double min = unDef, max = unDef;

        validNum = 0;
        for (i = 0; i < S.size(); i++) {
            if (!(doubleEquals(S.get(i), unDef))) {
                validNum++;
                if (validNum == 1) {
                    min = S.get(i);
                    max = min;
                } else {
                    if (S.get(i) < min) {
                        min = S.get(i);
                    }
                    if (S.get(i) > max) {
                        max = S.get(i);
                    }
                }
            }
        }

        return new double[]{min, max};
    }
    
    /**
     * Get mininum and maximum values
     *
     * @param S Data list
     * @return Minimum and Maximum data array
     */
    public static double[] getMinMaxValue(List S) {
        double min = 0, max = 0, v;
        for (int i = 0; i < S.size(); i++) {
            v = (double)S.get(i);
            if (i == 0){
                min = v;
                max = v;
            } else {
                if (min > v)
                    min = v;
                if (max < v)
                    max = v;
            }
        }

        return new double[]{min, max};
    }
    
    /**
     * Get mininum and maximum values
     *
     * @param S Data list
     * @return Minimum and Maximum data array
     */
    public static int[] getMinMaxInt(List S) {
        int min = 0, max = 0, v;
        for (int i = 0; i < S.size(); i++) {
            v = (int)S.get(i);
            if (i == 0){
                min = v;
                max = v;
            } else {
                if (min > v)
                    min = v;
                if (max < v)
                    max = v;
            }
        }

        return new int[]{min, max};
    }

    /**
     * Determine if a string is digital
     *
     * @param strNumber the string
     * @return Boolean
     */
    public static boolean isNumeric(String strNumber) {
        try {
            Double.parseDouble(strNumber);
            return true;
        } catch (NumberFormatException e) {
            return false;
        }
    }

    /**
     * Array reverse
     *
     * @param values Double array
     */
    public static void arrayReverse(double[] values) {
        int left = 0;          // index of leftmost element
        int right = values.length - 1; // index of rightmost element

        while (left < right) {
            // exchange the left and right elements
            double temp = values[left];
            values[left] = values[right];
            values[right] = temp;

            // move the bounds toward the center
            left++;
            right--;
        }
    }

    /**
     * Array reverse
     *
     * @param values Object array
     */
    public static void arrayReverse(Object[] values) {
        int left = 0;          // index of leftmost element
        int right = values.length - 1; // index of rightmost element

        while (left < right) {
            // exchange the left and right elements
            Object temp = values[left];
            values[left] = values[right];
            values[right] = temp;

            // move the bounds toward the center
            left++;
            right--;
        }
    }

    /**
     * Get min, max of an array
     *
     * @param values array
     * @return Min, max
     */
    public static double[] arrayMinMax(double[] values) {
        double min = values[0];
        double max = values[0];

        for (double value : values) {
            min = Math.min(value, min);
            max = Math.max(value, max);
        }

        return new double[]{min, max};
    }

    /**
     * Get min, max of an array
     *
     * @param values array
     * @return Min, max
     */
    public static double[] arrayMinMax(Double[] values) {
        double min = values[0];
        double max = values[0];

        for (double value : values) {
            min = Math.min(value, min);
            max = Math.max(value, max);
        }

        return new double[]{min, max};
    }

    /**
     * Determine if a point is in a rectangel
     *
     * @param aP The point
     * @param aRect The rectangel
     * @return Boolean
     */
    public static boolean pointInRectangle(Point aP, Rectangle aRect) {
        if (aP.x > aRect.x && aP.x < aRect.x + aRect.width && aP.y > aRect.y && aP.y < aRect.y + aRect.height) {
            return true;
        } else {
            return false;
        }
    }

    /**
     * Judge if a rectangle include another
     *
     * @param aRect a rectangle
     * @param bRect b rectangle
     * @return Boolean
     */
    public static boolean isInclude(Rectangle aRect, Rectangle bRect) {
        if (aRect.width >= bRect.width && aRect.height >= bRect.height) {
            if (aRect.x <= bRect.x && (aRect.x + aRect.width) >= (bRect.x + bRect.width)
                    && aRect.y <= bRect.y && (aRect.y + aRect.height) >= (bRect.y + bRect.height)) {
                return true;
            } else {
                return false;
            }
        } else {
            return false;
        }
    }

    /**
     * Calculate ellipse coordinate by angle
     *
     * @param x0 Center x
     * @param y0 Center y
     * @param a Major semi axis
     * @param b Minor semi axis
     * @param angle Angle
     * @return Coordinate
     */
    public static Point.Float calEllipseCoordByAngle(double x0, double y0, double a, double b, double angle) {
        double dx, dy;
        dx = Math.sqrt((a * a * b * b) / (b * b + a * a * Math.tan(angle) * Math.tan(angle)));
        dy = dx * Math.tan(angle);

        double x, y;
        if (angle <= Math.PI / 2) {
            x = x0 + dx;
            y = y0 + dy;
        } else if (angle <= Math.PI) {
            x = x0 - dx;
            y = y0 - dy;
        } else if (angle <= Math.PI * 1.5) {
            x = x0 - dx;
            y = y0 - dy;
        } else {
            x = x0 + dx;
            y = y0 + dy;
        }

        Point.Float aP = new Point.Float((float) x, (float) y);
        return aP;
    }

    /**
     * Get decimal number of a double data for ToString() format
     *
     * @param aData Data
     * @return Decimal number
     */
    public static int getDecimalNum(double aData) {
        if (aData - (int) aData == 0) {
            return 0;
        }

        double v = aData * 10;
        int dNum = 1;
        while (v - (int) v != 0) {
            if (dNum > 5) {
                break;
            }
            v = v * 10;
            dNum += 1;
        }

        return dNum;
    }

    /**
     * Get decimal number of a double data for ToString() format
     *
     * @param aData Data
     * @return Decimal number
     */
    public static int getDecimalNum_back(double aData) {
        if (aData - (int) aData == 0) {
            return 0;
        }

        int dNum;
        int aE = (int) Math.floor(Math.log10(aData));

        if (aE >= 0) {
            dNum = 2;
        } else {
            dNum = Math.abs(aE);
        }

        return dNum;
    }

    /**
     * Longitude distance
     *
     * @param lon1 Longitude 1
     * @param lon2 Longitude 2
     * @return Longitude distance
     */
    public static float lonDistance(float lon1, float lon2) {
        if (Math.abs(lon1 - lon2) > 180) {
            if (lon1 > lon2) {
                lon2 += 360;
            } else {
                lon1 += 360;
            }
        }

        return Math.abs(lon1 - lon2);
    }

    /**
     * Add longitude
     *
     * @param lon1 Longitude 1
     * @param delta Delta
     * @return Longitude
     */
    public static float lonAdd(float lon1, float delta) {
        float lon = lon1 + delta;
        if (lon > 180) {
            lon -= 360;
        }
        if (lon < -180) {
            lon += 360;
        }

        return lon;
    }

    /**
     * Get value from one dimension double array by index
     *
     * @param data Data
     * @param idx Index
     * @return Value
     */
    public static double getValue(double[] data, float idx) {
        double v = data[0];
        if (idx == 0) {
            return v;
        }

        for (int i = 1; i < data.length; i++) {
            if (idx == i) {
                v = data[i];
                break;
            } else if (idx < i) {
                v = data[i - 1] + (data[i] - data[i - 1]) * (idx - (i - 1));
                break;
            }
        }
        return v;
    }

    /**
     * Create values by interval
     *
     * @param min Miminum value
     * @param max Maximum value
     * @param interval Interval value
     * @return Value array
     */
    public static double[] getIntervalValues(double min, double max, double interval) {
        double[] cValues;
        min = BigDecimalUtil.add(min, interval);
        double mod = BigDecimalUtil.mod(min, interval);
        min = BigDecimalUtil.sub(min, mod);
        int cNum = (int) ((max - min) / interval) + 1;
        int i;

        cValues = new double[cNum];
        for (i = 0; i < cNum; i++) {
            cValues[i] = BigDecimalUtil.add(min, BigDecimalUtil.mul(i, interval));
        }

        return cValues;
    }

    /**
     * Get interval values
     *
     * @param min Minimum value
     * @param max Maximum value
     * @param n Level number
     * @return Values
     */
    public static double[] getIntervalValues(double min, double max, int n) {
        int aD, aE;
        double range;
        String eStr;

        range = BigDecimalUtil.sub(max, min);
        if (range == 0.0) {
            return new double[]{min};
        }

        eStr = String.format("%1$E", range);
        aD = Integer.parseInt(eStr.substring(0, 1));
        aE = (int) Math.floor(Math.log10(range));
        while (n > aD) {
            aD = aD * 10;
            aE = aE - 1;
        }
        double interval = BigDecimalUtil.mul((int) (aD / n), Math.pow(10, aE));

        return getIntervalValues(min, max, interval);
    }

    /**
     * Create contour values by minimum and maximum values
     *
     * @param min Minimum value
     * @param max Maximum value
     * @return Contour values
     */
    public static double[] getIntervalValues(double min, double max) {
        return (double[]) getIntervalValues(min, max, false).get(0);
    }

    /**
     * Create contour values by minimum and maximum values
     *
     * @param min Minimum value
     * @param max Maximum value
     * @return Contour values
     */
    public static List<Object> getIntervalValues1(double min, double max) {
        return getIntervalValues(min, max, false);
    }

    /**
     * Create contour values by minimum and maximum values
     *
     * @param min Minimum value
     * @param max Maximum value
     * @param isExtend If extend values
     * @return Contour values
     */
    public static List<Object> getIntervalValues(double min, double max, boolean isExtend) {
        int i, cNum, aD, aE;
        double cDelt, range, newMin;
        String eStr;
        List<Object> r = new ArrayList<>();

        range = BigDecimalUtil.sub(max, min);
        if (range == 0.0) {
            r.add(new double[]{min});
            r.add(0.0);
            return r;
        } else if (range < 0) {
            range = -range;
            double temp = min;
            min = max;
            max = temp;
        }

        eStr = String.format("%1$E", range);
        aD = Integer.parseInt(eStr.substring(0, 1));
        aE = (int) Math.floor(Math.log10(range));
//        int idx = eStr.indexOf("E");
//        if (idx < 0) {
//            aE = 0;
//        } else {
//            aE = Integer.parseInt(eStr.substring(eStr.indexOf("E") + 1));
//        }
        if (aD > 5) {
            //cDelt = Math.pow(10, aE);
            cDelt = BigDecimalUtil.pow(10, aE);
            cNum = aD;
            //newMin = Convert.ToInt32((min + cDelt) / Math.Pow(10, aE)) * Math.Pow(10, aE);
            //newMin = (int) (min / cDelt + 1) * cDelt;
        } else if (aD == 5) {
            //cDelt = aD * Math.pow(10, aE - 1);
            cDelt = aD * BigDecimalUtil.pow(10, aE - 1);
            cNum = 10;
            //newMin = Convert.ToInt32((min + cDelt) / Math.Pow(10, aE)) * Math.Pow(10, aE);
            //newMin = (int) (min / cDelt + 1) * cDelt;
            cNum++;
        } else {
            //cDelt = aD * Math.pow(10, aE - 1);
            double cd = BigDecimalUtil.pow(10, aE - 1);
            //cDelt = BigDecimalUtil.mul(aD, cDelt);
            cDelt = BigDecimalUtil.mul(5, cd);
            cNum = (int) (range / cDelt);
            if (cNum < 5) {
                cDelt = BigDecimalUtil.mul(2, cd);
                cNum = (int) (range / cDelt);
                if (cNum < 5) {
                    cDelt = BigDecimalUtil.mul(1, cd);
                    cNum = (int) (range / cDelt);
                }
            }
            //newMin = Convert.ToInt32((min + cDelt) / Math.Pow(10, aE - 1)) * Math.Pow(10, aE - 1);
            //newMin = (int) (min / cDelt + 1) * cDelt;            
        }
        int temp = (int) (min / cDelt + 1);
        newMin = BigDecimalUtil.mul(temp, cDelt);
        if (newMin - min >= cDelt) {
            newMin = BigDecimalUtil.sub(newMin, cDelt);
            cNum += 1;
        }

        if (newMin + (cNum - 1) * cDelt > max) {
            cNum -= 1;
        } else if (newMin + (cNum - 1) * cDelt + cDelt < max) {
            cNum += 1;
        }

        //Get values
        List<Double> values = new ArrayList<>();
        double v;
        for (i = 0; i < cNum; i++) {
            v = BigDecimalUtil.add(newMin, BigDecimalUtil.mul(i, cDelt));
            if (v >= min && v <= max)
                values.add(v);
        }

        //Extend values
        if (isExtend) {
            if (values.get(0) > min) {
                values.add(0, BigDecimalUtil.sub(newMin, cDelt));
            }
            if (values.get(values.size() - 1) < max) {
                values.add(BigDecimalUtil.add(values.get(values.size() - 1), cDelt));
            }
        }

        double[] cValues = new double[values.size()];
        for (i = 0; i < values.size(); i++) {
            cValues[i] = values.get(i);
        }

        r.add(cValues);
        r.add(cDelt);
        return r;
    }

    /**
     * Create log interval values by minimum and maximum values
     *
     * @param min Minimum value
     * @param max Maximum value
     * @return Interval values
     */
    public static double[] getIntervalValues_Log(double min, double max) {
        int i, v;
        int minE = (int) Math.floor(Math.log10(min));
        int maxE = (int) Math.ceil(Math.log10(max));
        if (min == 0) {
            minE = maxE - 2;
        }
        if (max == 0) {
            maxE = minE + 2;
        }

        double[] cValues = new double[maxE - minE + 1];
        i = 0;
        for (v = minE; v <= maxE; v++) {
            cValues[i] = Math.pow(10, v);
            i++;
        }

        return cValues;
    }

    /**
     * Create log interval values by minimum and maximum values
     *
     * @param min Minimum value
     * @param max Maximum value
     * @return Interval values
     */
    public static double[] getIntervalValues_Log_bak(double min, double max) {
        int i, v;
        int minE = (int) Math.floor(Math.log10(min));
        int maxE = (int) Math.ceil(Math.log10(max));
        if (min == 0) {
            minE = maxE - 2;
        }
        if (max == 0) {
            maxE = minE + 2;
        }

        List<Double> values = new ArrayList<>();
        double vv;
        for (v = minE; v <= maxE; v++) {
            vv = Math.pow(10, v);
            if (vv >= min && vv <= max) {
                values.add(vv);
            }
        }
        double[] cValues = new double[values.size()];
        for (i = 0; i < values.size(); i++) {
            cValues[i] = values.get(i);
        }

        return cValues;
    }

    /**
     * Convert cartesian to polar coordinate
     *
     * @param x X
     * @param y Y
     * @return Angle and radius
     */
    public static double[] cartesianToPolar(double x, double y) {
        double r;     // Radius
        double B;     // Angle in radians
        r = Math.hypot(x, y);
        B = Math.atan2(y, x);
//        if (y >= 0) {
//            if (x == 0) {
//                B = Math.PI / 2;// 90°
//            } else {
//                B = Math.atan(y / x);
//            }
//        } else if (x == 0) {
//            B = 3 * Math.PI / 2;// 270°
//        } else {
//            B = Math.atan(y / x);
//        }
        return new double[]{B, r};
    }

    /**
     * Convert poar to cartesian coordinate
     *
     * @param r Radius
     * @param B Angle in radians
     * @return X and y in cartesian coordinate
     */
    public static double[] polarToCartesian(double B, double r) {
        double x = Math.cos(B) * r;
        double y = Math.sin(B) * r;

        return new double[]{x, y};
    }
}
