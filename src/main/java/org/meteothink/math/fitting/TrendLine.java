/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.meteothink.math.fitting;

import org.meteothink.ndarray.Array;

/**
 *
 * @author Yaqiang Wang
 */
public interface TrendLine {
    public void setValues(Array y, Array x); // y ~ f(x)
    public double predict(double x); // get a predicted y for a given x
}
