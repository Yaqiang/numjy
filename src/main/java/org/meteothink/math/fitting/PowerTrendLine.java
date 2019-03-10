/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.meteothink.math.fitting;

/**
 *
 * @author Yaqiang Wang
 */
public class PowerTrendLine extends OLSTrendLine {
    
    @Override
    protected double[] xVector(double x) {
        return new double[]{1,Math.log(x)};
    }

    @Override
    protected boolean logY() {return true;}
}
