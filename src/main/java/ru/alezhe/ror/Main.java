/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ru.alezhe.ror;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.ParseException;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import ru.alezhe.ror.GarchModel.GarchApproximatedTimeSeries;

/**
 *
 * @author Aleksandr
 */
public class Main {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws IOException, ParseException {
//        develop1();
//        develop2();
        testScreening(new File("D:\\workspace\\GARCHA\\pool2"));
    }

    private static void develop1() throws IOException, ParseException {
        ObservedTimeSeries ts = ObservedTimeSeries.readFinamFile(new File("D:\\workspace\\GARCHA\\SBER_000101_170715 (1).txt"), "yyyyMMdd", "CLOSE");
        ts = ts.getRelativeLogarithmicSeries();

        GarchModel gm = new GarchModel(1, 2);
        int windowSize = 60;

//        gm.modelGARCHPredictor(ts, 60, 60);
//        Map<Integer, GarchModel.ModeledGARCHSeries> series = gm.modelGARCHPredictors(ts, windowSize);
        Map<Integer, GarchModel.ModeledGARCHSeries> series = gm.modelProgressiveGARCHPredictors(ts, windowSize);

        for (Map.Entry<Integer, GarchModel.ModeledGARCHSeries> entry : series.entrySet()) {
            Integer day = entry.getKey();
            GarchModel.ModeledGARCHSeries model = entry.getValue();

            System.out.print(day + "\t" + ts.getDate(day) + "\t");
            if (model != null) {
                double var = model.calcVaR(0.95);
                System.out.print(model.getNextStepVariance() + "\t" + var + "\t");
                if (day + 1 < ts.getLength()) {
                    double nextDay = ts.getValue(day + 1);
                    System.out.print(nextDay + "\t" + (nextDay < var ? "failed" : ""));
                }
            }
            System.out.println();
        }

        int h = 0;
    }

    private static void develop2() throws IOException, ParseException {
        ObservedTimeSeries t = ObservedTimeSeries.readFinamFile(new File("D:\\workspace\\GARCHA\\pool1\\SBER_000101_170715 (1).txt"), "yyyyMMdd", "CLOSE");
        ObservedTimeSeries ts = t.getRelativeLogarithmicSeries();
        GarchModel gm = new GarchModel(2, 2);
        int windowSize = 300;

        int fails = 0;
        final GarchApproximatedTimeSeries garchTs = gm.approximateTimeSeries(ts, windowSize);
        for (int timePoint = windowSize; timePoint < ts.getLength(); timePoint++) {
            double variance = garchTs.getVariance(timePoint);
            double value = garchTs.getValue(timePoint);
            double vaR = garchTs.getVaR(timePoint, 0.95);
            boolean failed = value < vaR;
            fails += failed ? 1 : 0;

            System.out.println(timePoint + "\t" + ts.getDate(timePoint) + "\t" + ts.getValue(timePoint) + "\t" + variance + "\t" + vaR + "\t" + value + "\t" + (failed ? "failed" : ""));
        }
        System.out.println(fails + " fails");

        int q = 9;

    }

    private static void testScreening(File srcFolder) throws IOException, ParseException {
        final File dstFolder = new File(srcFolder.getParentFile(), srcFolder.getName() + "_test");
        dstFolder.mkdir();

        final List<GarchModel> models = Arrays.asList(new GarchModel[]{
            new GarchModel(1, 1),
            new GarchModel(1, 2),
            new GarchModel(2, 1),
            new GarchModel(2, 2),});

        double alpha = 0.95;

        for (File file : srcFolder.listFiles()) {
            final String name = file.getName().replaceFirst("[.][^.]+$", "");
            final ObservedTimeSeries observed = ObservedTimeSeries.readFinamFile(file, "yyyyMMdd", "CLOSE");
            final ObservedTimeSeries relative = observed.getRelativeLogarithmicSeries();
            for (GarchModel model : models) {
                for (int windowSize = 50; windowSize < 251; windowSize += 50) {
                    File outFile = new File(dstFolder, String.format("%s_%s_%d_%.2f.tsv", name, model.getId(), windowSize, alpha));
                    testSeries(outFile, relative, model, windowSize, alpha);
                }
            }
        }
    }

    private static void testSeries(File outFile, ObservedTimeSeries ts, GarchModel model, int windowSize, double alpha) throws IOException {
        System.out.println(outFile.getName());
        final GarchApproximatedTimeSeries garchTs = model.approximateTimeSeries(ts, windowSize);

        int fails = 0;
        try (FileWriter writer = new FileWriter(outFile)) {
            writer.append(String.format("%s\t%s\t%s\t%s\t%s\n", "Time point", "Date", "Value", "Predicted Variance", "VaR"));
            for (int timePoint = windowSize; timePoint < ts.getLength(); timePoint++) {
                double variance = garchTs.getVariance(timePoint);
                double value = garchTs.getValue(timePoint);
                double vaR = garchTs.getVaR(timePoint, alpha);
                boolean failed = value < vaR;
                fails += failed ? 1 : 0;

                writer.append(String.format("%d\t%s\t%f\t%f\t%f", timePoint, ts.getDate(timePoint).toString(), value, variance, vaR));
                writer.append(failed ? "\tfailed\n" : "\n");
            }
        }
        System.out.println(outFile.getName() + "\t" + fails + " fails");
    }

}
