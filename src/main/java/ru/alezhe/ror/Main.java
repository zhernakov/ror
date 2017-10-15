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
import ru.alezhe.ror.GarchModel.MarGarchApproximatedTimeSeries;

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
        develop2();
//        testScreening(new File("D:\\workspace\\GARCHA\\pool2"));
    }

    private static void develop1() throws IOException, ParseException {
        ObservedTimeSeries ts = ObservedTimeSeries.readFinamFile(new File("D:\\workspace\\GARCHA\\SBER_000101_170715 (1).txt"), "yyyyMMdd", "CLOSE");
        ts = ts.getRelativeLogarithmicSeries();

        GarchModel gm = new GarchModel(1, 2);
        int windowSize = 60;

        int h = 0;
    }

    private static void develop2() throws IOException, ParseException {
//        ObservedTimeSeries t = ObservedTimeSeries.readFinamFile(new File("D:\\workspace\\GARCHA\\pool1\\SBER_000101_170715 (1).txt"), "yyyyMMdd", "CLOSE");
        ObservedTimeSeries t = ObservedTimeSeries.readFinamFile(new File("D:\\workspace\\GARCHA\\pool2\\DIXY_000101_170715.txt"), "yyyyMMdd", "CLOSE");
        ObservedTimeSeries ts = t.getRelativeLogarithmicSeries();
        GarchModel gm = new GarchModel(1, 2);
        int windowSize = 100;
        double alpha = 0.95;

        int fails = 0;
        int marFails = 0;
        
//        GarchApproximatedTimeSeries marTs = gm.approximateTimeSeries(ts, windowSize);
        MarGarchApproximatedTimeSeries marTs = gm.approximateMarTimeSeries(ts, windowSize, alpha);
        for (int timePoint = windowSize; timePoint < ts.getLength(); timePoint++) {
            double variance = marTs.getVariance(timePoint);
            double value = marTs.getValue(timePoint);
            double vaR = marTs.getVaR(timePoint, 0.95);
            double maR = marTs.getMaR(timePoint);
            boolean failed = value < vaR;
            fails += failed ? 1 : 0;
            marFails += failed && value < maR ? 1 : 0;

            System.out.println(timePoint + "\t" + ts.getDate(timePoint) + "\t" + value + "\t" + variance + "\t" + vaR + "\t" + (failed ? "failed" : "") + "\t" + maR);
        }
        System.out.printf("%d/%d\n", fails, marFails);
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
