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
package org.meteothink.ndarray;

import java.util.ArrayList;
import java.util.List;
import org.meteothink.util.BigDecimalUtil;

/**
 * Template
 *
 * @author Yaqiang Wang
 */
public class Dimension {
    // <editor-fold desc="Variables">

    //private ucar.nc2.Dimension _ncDimension = null;
    private String name;
    private DimensionType _dimType;
    private List<Double> _dimValue = new ArrayList<>();
    private int _dimId;
    private int length = 1;
    private boolean unlimited = false;
    private boolean variableLength = false;
    private boolean shared = true;
    private boolean reverse = false;

    /**
     * Constructor
     */
    public Dimension() {
        this("null", 1);
    }

    /**
     * Constructor
     *
     * @param name Name
     * @param len Length
     */
    public Dimension(String name, int len) {
        this.name = name;
        this.length = len;
        _dimType = DimensionType.Other;
        _dimValue = new ArrayList<>();
    }

    /**
     * Constructor
     *
     * @param dimType Dimension type
     */
    public Dimension(DimensionType dimType) {
        this(dimType.toString(), 1, dimType);
    }

    /**
     * Constructor
     *
     * @param name Name
     * @param len Length
     * @param dimType Dimension type
     */
    public Dimension(String name, int len, DimensionType dimType) {
        this(name, len);
        _dimType = dimType;
        _dimValue = new ArrayList<>();
    }

    /**
     * Constructor
     *
     * @param name Name
     * @param len Length
     * @param dimType Dimension type
     * @param min Minimum value
     * @param delta Delta value
     * @param num value number
     */
    public Dimension(String name, int len, DimensionType dimType, double min, double delta, int num) {
        this(name, len);
        _dimType = dimType;
        _dimValue = new ArrayList<>();
        for (int i = 0; i < num; i++) {
            _dimValue.add(min + delta * i);
        }
    }
    // </editor-fold>
    // <editor-fold desc="Get Set Methods">
    /**
     * Get short name
     * @return Short name
     */
    public String getShortName() {
        return this.name;
    }
    
    /**
     * Set short name
     * @param value Short name
     */
    public void setShortName(String value) {
        this.name = value;
    }
    
    /**
     * Get length
     * @return Length
     */
    public int getLength() {
        return this.length;
    }
   
    /**
     * Set dimension length
     *
     * @param value Dimension length
     */
    public void setLength(int value) {
        if (value <= 0)
            return;
                   
        this.length = value;
        if (this._dimValue == null)
            this._dimValue = new ArrayList<>();
        if (this._dimValue.size() != value){
            this._dimValue.clear();
            for (int i = 0; i < value; i++) {
                this._dimValue.add(Double.valueOf(i));
            }
        }
    }

    /**
     * Get dimension type
     *
     * @return Dimension type
     */
    public DimensionType getDimType() {
        return _dimType;
    }

    /**
     * Set dimension type
     *
     * @param value Dimension type
     */
    public void setDimType(DimensionType value) {
        _dimType = value;
    }

    /**
     * Get dimension values
     *
     * @return Dimension values
     */
    public List<Double> getDimValue() {
        return _dimValue;
    }
    
    /**
     * Get dimension value array
     * @return dimension value array
     */
    public Array getDimArray() {
        int n = this.getLength();
        Array r = Array.factory(DataType.DOUBLE, new int[]{n});
        for (int i = 0; i < n; i++) {
            r.setDouble(i, _dimValue.get(i));
        }
        
        return r;
    }
    
    /**
     * Get dimension value by index
     * @param idx index
     * @return Dimension value
     */
    public double getDimValue(int idx) {
        return this._dimValue.get(idx);
    }

    /**
     * Get dimension identifer
     *
     * @return Dimension identifer
     */
    public int getDimId() {
        return _dimId;
    }

    public void setDimId(int value) {
        _dimId = value;
    }

    /**
     * Get if values are reverse (in descending order)
     *
     * @return Boolean
     */
    public boolean isReverse() {
        return this.reverse;
    }

    /**
     * Set if values are reverse
     *
     * @param value Boolean
     */
    public void setReverse(boolean value) {
        this.reverse = value;
    }
    
    /**
     * Get is unlimited or not
     * @return Boolean
     */
    public boolean isUnlimited() {
        return this.unlimited;
    }
    
    /**
     * Set unlimited or not
     * @param value Boolean
     */
    public void setUnlimited(boolean value) {
        this.unlimited = value;
    }
    
    /**
     * Get is shared or not
     * @return Boolean
     */
    public boolean isShared() {
        return this.shared;
    }
    
    /**
     * Set is shared or not
     * @param value Boolean
     */
    public void setShared(boolean value) {
        this.shared = value;
    }
    
    /**
     * Get is variable length or not
     * @return Boolean
     */
    public boolean isVariableLength() {
        return this.variableLength;
    }
    
    /**
     * Set is variable length or not
     * @param value 
     */
    public void setVariableLength(boolean value) {
        this.variableLength = value;
    }
    // </editor-fold>
    // <editor-fold desc="Methods">

    /**
     * Determine if two dimensions equals
     *
     * @param aDim The other dimension
     * @return If equals
     */
    public boolean equals(Dimension aDim) {
        if (!this.getShortName().equals(aDim.getShortName())) {
            return false;
        }
        if (_dimType != aDim.getDimType()) {
            return false;
        }
        return this.getLength() == aDim.getLength();
    }

    /**
     * Get dimension value array
     *
     * @return Value array
     */
    public double[] getValues() {
        int len = this.getLength();
        double[] values = new double[len];
        for (int i = 0; i < len; i++) {
            values[i] = _dimValue.get(i);
        }

        return values;
    }

    /**
     * Set dimension values
     *
     * @param values Values
     */
    public void setValues(List<Double> values) {
        _dimValue = values;
        this.setLength(_dimValue.size());
    }

    /**
     * Set dimension values
     *
     * @param values Values
     */
    public void setDimValues(List<Number> values) {
        _dimValue = new ArrayList<>();
        for (Number v : values) {
            _dimValue.add(v.doubleValue());
        }
        this.setLength(_dimValue.size());
    }

    /**
     * Set dimension values
     *
     * @param values Values
     */
    public void setValues(double[] values) {
        _dimValue = new ArrayList<>();
        for (double v : values) {
            _dimValue.add(v);
        }
        this.setLength(_dimValue.size());
    }

    /**
     * Set dimension values
     *
     * @param values Values
     */
    public void setValues(float[] values) {
        _dimValue = new ArrayList<>();
        for (double v : values) {
            _dimValue.add(v);
        }
        this.setLength(_dimValue.size());
    }

    /**
     * Add a dimension value
     *
     * @param value The value
     */
    public void addValue(double value) {
        _dimValue.add(value);
        this.setLength(_dimValue.size());
    }

    /**
     * Get minimum dimension value
     *
     * @return Minimum dimension value
     */
    public double getMinValue() {
        return _dimValue.get(0);
    }

    /**
     * Get maximum dimension value
     *
     * @return Maximum dimension value
     */
    public double getMaxValue() {
        return _dimValue.get(_dimValue.size() - 1);
    }

    /**
     * Get delta value
     *
     * @return Delta value
     */
    public double getDeltaValue() {
        if (_dimValue.size() <= 1) {
            return 1;
        }

        return BigDecimalUtil.sub(_dimValue.get(1), _dimValue.get(0));
    }
    
    /**
     * Extract dimension
     * @param range The range
     * @return Result dimension
     */
    public Dimension extract(Range range) {
        return this.extract(range.first(), range.last(), range.stride());
    }

    /**
     * Extract dimension
     *
     * @param first First
     * @param last Last
     * @param stride Stride
     * @return Extracted dimension
     */
    public Dimension extract(int first, int last, int stride) {
        Dimension dim = new Dimension(this.getShortName(), this.getLength(), this._dimType);
        dim.setDimId(this._dimId);
        dim.setReverse(this.reverse);
        if (this._dimValue.size() > last) {
            List<Double> values = new ArrayList<>();
            int step = Math.abs(stride);
            if (this.reverse) {
                int ff = this.getLength() - last - 1;
                int ll = this.getLength() - first - 1;
                for (int i = ff; i <= ll; i += step) {
                    values.add(this._dimValue.get(i));
                }
            } else {
                for (int i = first; i <= last; i += step) {
                    values.add(this._dimValue.get(i));
                }
            }
            dim.setValues(values);
        }

        return dim;
    }

    /**
     * Extract dimension
     *
     * @param first First
     * @param last Last
     * @param stride Stride
     * @return Extracted dimension
     */
    public Dimension extract(double first, double last, double stride) {
        Dimension dim = new Dimension(this.getShortName(), this.getLength(), this._dimType);
        dim.setDimId(this._dimId);
        List<Double> values = new ArrayList<>();
        int idx;
        for (double v = first; v <= last; v += stride) {
            idx = this.getValueIndex(v);
            values.add(this._dimValue.get(idx));
        }
        dim.setValues(values);

        return dim;
    }
    
    /**
     * Extract dimension
     *
     * @param index Indices
     * @return Extracted dimension
     */
    public Dimension extract(List<Integer> index) {
        Dimension dim = new Dimension(this.getShortName(), this.getLength(), this._dimType);
        dim.setDimId(this._dimId);
        dim.setReverse(this.reverse);
        List<Double> values = new ArrayList<>();
            if (this.reverse) {
                for (int i = index.size() - 1; i <= 0; i--) {
                    values.add(this._dimValue.get(index.get(i)));
                }
            } else {
                for (int i = 0; i < index.size(); i++) {
                    values.add(this._dimValue.get(index.get(i)));
                }
            }
            dim.setValues(values);

        return dim;
    }

    /**
     * Get value index
     *
     * @param v Value
     * @return Index
     */
    public int getValueIndex(double v) {
        int idx = this.getLength() - 1;
        if (getDeltaValue() > 0) {
            for (int i = 0; i < this.getLength(); i++) {
                if (v <= this._dimValue.get(i)) {
                    if (i == 0)
                        idx = 0;
                    else {
                        if (this._dimValue.get(i) - v > v - this._dimValue.get(i - 1))
                            idx = i - 1;
                        else
                            idx = i;
                    }
                    break;
                }
            }
        } else {
            for (int i = 0; i < this.getLength(); i++) {
                if (v >= this._dimValue.get(i)) {
                    if (i == 0)
                        idx = 0;
                    else {
                        if (this._dimValue.get(i - 1) - v > v - this._dimValue.get(i))
                            idx = i;
                        else
                            idx = i - 1;
                    }
                    break;
                }
            }
        }
        if (this.reverse) {
            idx = this.getLength() - idx - 1;
        }

        return idx;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("Name: ").append(this.getShortName());
        sb.append("\n");
        sb.append("Min value: ").append(String.valueOf(this.getMinValue()));
        sb.append("\n");
        sb.append("Max value: ").append(String.valueOf(this.getMaxValue()));
        sb.append("\n");
        sb.append("Size: ").append(String.valueOf(this.getLength()));
        sb.append("\n");
        sb.append("Delta: ").append(String.valueOf(this.getDeltaValue()));

        return sb.toString();
    }
    // </editor-fold>
}
