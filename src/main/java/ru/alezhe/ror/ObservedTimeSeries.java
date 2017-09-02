/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ru.alezhe.ror;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.text.DateFormat;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Date;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.stream.IntStream;
import org.apache.commons.math3.stat.descriptive.moment.Variance;

/**
 *
 * @author Aleksandr
 */
public class ObservedTimeSeries {
    protected static final Variance VARIANCE = new Variance();
    
    protected final int n;
    protected final Date[] dates;
    protected final double[] values;

    protected ObservedTimeSeries(Date[] dates, double[] values) {
        n = dates.length;
        this.dates = dates;
        this.values = values;
    }

    public ObservedTimeSeries(ObservedTimeSeries series) {
        this(series.dates.clone(), series.values.clone());
    }
    
    public int getLength() {
        return values.length;
    }
    
    public double getVariance(int from, int to) {
        return VARIANCE.evaluate(values, from, to);
    }
    
    public Date getDate(int timePoint) {
        return dates[timePoint];
    }
    
    public double getValue(int timePoint) {
        return values[timePoint];
    }
    
    public double[] getValues(int from, int to) {
        return Arrays.copyOfRange(values, from, to);
    }
    
    //////////////
    public ObservedTimeSeries getRelativeLogarithmicSeries() {
        double[] newSeries = new double[values.length];
        newSeries[0] = 0;
        for (int i = 1; i < values.length; i++) {
            newSeries[i] = Math.log(values[i] / values[i - 1]);
        }
        return new ObservedTimeSeries(dates.clone(), newSeries);
    }
    
    //////////////
    public static ObservedTimeSeries readFinamFile(File file, String dateFormatString, String valueLable) throws IOException, ParseException {
        final LinkedList<Date> dates = new LinkedList<>();
        final LinkedList<Double> series = new LinkedList<>();

        String line;
        final DateFormat df = new SimpleDateFormat(dateFormatString);
        try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
            final String[] lables = reader.readLine().split("(\\s+)");
            final LinkedHashMap<String, Integer> map = IntStream.range(0, lables.length)
                    .collect(LinkedHashMap<String,Integer>::new , 
                            (m,v) -> {m.put(lables[v].replaceAll(">|<", ""),v);}, 
                            (m1,m2)-> m1.putAll(m2));
            final int dateInd = map.get("DATE");
            final int valInd = map.get(valueLable);
 
            while ((line = reader.readLine()) != null) {
                final String[] values = line.split("\\s+");
                dates.add(df.parse(values[dateInd]));
                series.add(Double.parseDouble(values[valInd]));
            }
        }
        
        int n = dates.size();
        return new ObservedTimeSeries(
                dates.toArray(new Date[n]), 
                series.stream().mapToDouble(Double::doubleValue).toArray());
    }
}
