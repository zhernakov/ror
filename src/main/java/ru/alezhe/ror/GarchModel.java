/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ru.alezhe.ror;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.stream.DoubleStream;
import org.apache.commons.math3.analysis.MultivariateFunction;
import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.exception.MathIllegalStateException;
import org.apache.commons.math3.optim.InitialGuess;
import org.apache.commons.math3.optim.MaxEval;
import org.apache.commons.math3.optim.PointValuePair;
import org.apache.commons.math3.optim.SimpleBounds;
import org.apache.commons.math3.optim.SimpleValueChecker;
import org.apache.commons.math3.optim.linear.LinearConstraint;
import org.apache.commons.math3.optim.linear.LinearConstraintSet;
import org.apache.commons.math3.optim.linear.Relationship;
import org.apache.commons.math3.optim.nonlinear.scalar.GoalType;
import org.apache.commons.math3.optim.nonlinear.scalar.MultivariateOptimizer;
import org.apache.commons.math3.optim.nonlinear.scalar.ObjectiveFunction;
import org.apache.commons.math3.optim.nonlinear.scalar.gradient.NonLinearConjugateGradientOptimizer;
import org.apache.commons.math3.optim.nonlinear.scalar.noderiv.BOBYQAOptimizer;
import org.apache.commons.math3.optim.nonlinear.scalar.noderiv.PowellOptimizer;
import org.apache.commons.math3.stat.descriptive.moment.Variance;

/**
 *
 * @author Aleksandr
 */
public class GarchModel {

    private static final NormalDistribution NID = new NormalDistribution();
    private static final Variance VARIANCE = new Variance();

    private final int p;
    private final int q;
//    private final int length;
    private final double[] guesses;
    private final double[] lowerBound;
    private final double[] upperBound;
//    private final ConvergenceChecker checker = new SimpleValueChecker(1e-13, 1e-13); 
    private final LinearConstraint constraint;
    
    private final String id;

    public GarchModel(int p, int q) {
        this.p = p;
        this.q = q;
//        this.length = Math.max(p, q);
        int n = p + q + 1;
        guesses = new double[n];
        guesses[0] = 0.0004;
        Arrays.fill(guesses, 1, p + 1, 0.2);
        Arrays.fill(guesses, p + 1, p + q + 1, 0.8);
        lowerBound = new double[n];
        Arrays.fill(lowerBound, 0.0001);
        upperBound = new double[n];
        Arrays.fill(upperBound, 1);
        double[] coef = new double[n];
        Arrays.fill(coef, 1);
        coef[0] = 0;
        constraint = new LinearConstraint(coef, Relationship.LEQ, 1);
        
        id = String.format("GARCH%d,%d", p, q);
    }

    public String getId() {
        return id;
    }

    public ModeledGARCHSeries modelGARCHPredictor(ObservedTimeSeries series, int windowSize, int forIndex) {
        MaxLikelihoodGARCHFunction lmfun = new MaxLikelihoodGARCHFunction(series.getValues(forIndex - windowSize, forIndex));
        MultivariateOptimizer optimizer = new BOBYQAOptimizer(p + q + 3);
//        MultivariateOptimizer optimizer = new NonLinearConjugateGradientOptimizer(
//                NonLinearConjugateGradientOptimizer.Formula.POLAK_RIBIERE, new SimpleValueChecker(0.0001, 0.00000001, 50000));
//        MultivariateOptimizer optimizer = new PowellOptimizer(1e-8, 1e-8);

        try {
            PointValuePair optResult = optimizer.optimize(
                    GoalType.MAXIMIZE,
                    new MaxEval(50000),
                    new ObjectiveFunction(lmfun),
                    new InitialGuess(guesses),
                    new SimpleBounds(lowerBound, upperBound),
                    new LinearConstraintSet(constraint)
            );
            double[] param = optResult.getPoint();
            ModeledGARCHSeries modeled = new ModeledGARCHSeries(
                    param,
                    lmfun.getSquaredValues(),
                    lmfun.getVariancesValues(param),
                    lmfun.getUnconditionalVariance());

            double mlVal = lmfun.value(param);
            return modeled;
        } catch (MathIllegalStateException e) {
        }

        return null;
    }

    public Map<Integer, ModeledGARCHSeries> modelGARCHPredictors(ObservedTimeSeries series, int windowSize) {
        Map<Integer, ModeledGARCHSeries> predictors = new LinkedHashMap<>();
        for (int i = windowSize; i < series.getLength(); i++) {
            predictors.put(i, modelGARCHPredictor(series, windowSize, i));
        }
        return predictors;
    }

    public Map<Integer, ModeledGARCHSeries> modelProgressiveGARCHPredictors(ObservedTimeSeries series, int windowSize) {
        Map<Integer, ModeledGARCHSeries> predictors = new LinkedHashMap<>();
        for (int i = windowSize; i < series.getLength() - 1; i++) {
            predictors.put(i, modelGARCHPredictor(series, i, i));
        }
        return predictors;
    }

    //////////////////////////
    public GarchApproximatedTimeSeries approximateTimeSeries(ObservedTimeSeries ts, int windowSize) {
        return new GarchApproximatedTimeSeries(ts, windowSize);
    }

    //////////////////////////
    public class GarchApproximation {

        private final double omega;
        private final double[] aa;
        private final double[] bb;

        public GarchApproximation(double omega, double[] aa, double[] bb) {
            this.omega = omega;
            this.aa = aa;
            this.bb = bb;
        }

        public GarchApproximation(double[] param) {
            this(
                    param[0],
                    Arrays.copyOfRange(param, 1, p + 1),
                    Arrays.copyOfRange(param, p + 1, p + q + 1));
        }

        public double getVariance(int valuePoint, double[] values, int variancePoint, double[] variances) {
            int index;
            double variance = omega;
            for (int i = 0; i < p; i++) {
                if ((index = valuePoint - i - 1) >= 0) {
                    variance += aa[i] * Math.pow(values[index], 2);
                }
            }
            for (int j = 0; j < q; j++) {
                if ((index = variancePoint - j - 1) >= 0) {
                    variance += bb[j] * variances[index];
                }
            }
            return variance;
        }

        public double[] getParameters() {
            final double[] params = new double[p + q + 1];
            params[0] = omega;
            System.arraycopy(aa, 0, params, 1, p);
            System.arraycopy(bb, 0, params, 1 + p, q);
            return params;
        }
    }

    public class GarchApproximatedTimeSeries extends ObservedTimeSeries {

        private final GarchApproximation[] approximations;
        private final double[][] variances;
        private final int windowSize;

        public GarchApproximatedTimeSeries(ObservedTimeSeries series, int windowSize) {
            super(series);
            this.approximations = new GarchApproximation[n];
            this.variances = new double[n][];
            this.variances[0] = new double[q];
            Arrays.fill(variances[0], VARIANCE.evaluate(values));
            this.windowSize = windowSize;
            approximate();
        }

        private void approximate() {
            for (int i = 1; i < n; i++) {
                GarchApproximation aprx = fitGarchParameters(i);
                if (aprx == null) {
                    aprx = approximations[i - 1];
                    System.err.println("failed to optimize at timepoint " + i);
                }
                System.out.println(i);
                approximations[i] = aprx;
                if (aprx != null) {
                    double[] fitVars = calcVariances(aprx, i);
                    variances[i] = Arrays.copyOfRange(fitVars, fitVars.length - q, fitVars.length);
                } else {
                    variances[i] = variances[i - 1];
                }
            }
        }

        private void someCheck(int timePoint, GarchApproximation aprx) {

        }

        private double[] calcVariances(GarchApproximation aprx, int timePoint) {
            final int realWindowSize = Math.min(windowSize, timePoint);
            final int first = timePoint - realWindowSize;
            final double[] vars = new double[realWindowSize + q];
            System.arraycopy(variances[first], 0, vars, 0, q);
            for (int i = 0; i < realWindowSize; i++) {
                vars[i + q] = aprx.getVariance(first + i, values, i + q, vars);
            }
            return vars;
        }

        private GarchApproximation fitGarchParameters(int i) {
            MultivariateOptimizer optimizer = new BOBYQAOptimizer((p + q + 1) * 1 + (2));
            GarchMaxLikelihoodFunction lmfun = new GarchMaxLikelihoodFunction(i);
            try {
                PointValuePair optResult = optimizer.optimize(
                        GoalType.MAXIMIZE,
                        new MaxEval(100000),
                        new ObjectiveFunction(lmfun),
                        //                        new InitialGuess(i < 2 ? guesses : approximations[i - 1].getParameters()),
                        new InitialGuess(guesses),
                        new SimpleBounds(lowerBound, upperBound),
                        new LinearConstraintSet(constraint)
                );
                double[] param = optResult.getPoint();
                GarchApproximation apr = new GarchApproximation(param);
                return apr;
            } catch (MathIllegalStateException e) {
//                optimizer.
                return null;
//                int h = 9;
            }
        }

        ////////////////////////
        public double getVariance(int timePoint) {
            GarchApproximation apr = approximations[timePoint];
            return apr.getVariance(timePoint, values, q, variances[timePoint]);
        }

        public double getVaR(int timePoint, double prob) {
            double quantile = NID.inverseCumulativeProbability(1 - prob);
            double variance = getVariance(timePoint);
            return quantile * Math.sqrt(variance);
        }

        ////////////////////////
        private class GarchMaxLikelihoodFunction implements MultivariateFunction {

            private final int timePoint;

            public GarchMaxLikelihoodFunction(int timePoint) {
                this.timePoint = timePoint;
            }

            @Override
            public double value(double[] params) {
                if (timePoint == 60) {
                    int y = 0;
                }

                final GarchApproximation apr = new GarchApproximation(params);
                final double[] vars = calcVariances(apr, timePoint);
                final int realWindowSize = Math.min(windowSize, timePoint);
                final int first = timePoint - realWindowSize;

                double lml = 0;
                for (int i = first; i < timePoint; i++) {
                    double variance = vars[i - first + q];
                    double lmlp = Math.log(1 / Math.sqrt(variance * 2 * Math.PI)) - Math.pow(values[i], 2) / variance / 2;
                    lml += lmlp;
                }
//                for (int i = Math.max(0, timePoint - windowSize); i < timePoint; i++) {
//                    double variance = apr.getVariance(i, values, q, variances[i]);
//                    double lmlp = Math.log(1 / Math.sqrt(variance * 2 * Math.PI)) - Math.pow(values[i], 2) / variance / 2;
//                    lml += lmlp;
//                }
                return lml;
            }

        }
    }

    //////////////////////////
    public class ProgressiveModeledGARCHseries extends ObservedTimeSeries {

        private final Map<Integer, ModeledGARCHSeries> modeles = new LinkedHashMap<>();

        public ProgressiveModeledGARCHseries(ObservedTimeSeries series, int startWindowSize) {
            super(series);
        }

    }

    //////////////////////////
    public class ModeledGARCHSeries implements Predictor {

        private final double omega;
        private final double[] aa;
        private final double[] bb;
        private final double[] squaredValues;
        private final double[] variances;
        private final double uncVariance;
        private double nextStepVariance;

        public ModeledGARCHSeries(double[] param, double[] squaredValues, double[] variances, double uncVariance) {
            this.omega = param[0];
            this.aa = Arrays.copyOfRange(param, 1, p + 1);
            this.bb = Arrays.copyOfRange(param, p + 1, p + q + 1);
            this.squaredValues = squaredValues;
            this.variances = variances;
            this.uncVariance = uncVariance;
            calcNextStepVariance();
        }

        private void calcNextStepVariance() {
            nextStepVariance = omega;
            for (int i = 0; i < p; i++) {
                nextStepVariance += aa[i] * squaredValues[i];
            }
            for (int i = 0; i < q; i++) {
                nextStepVariance += bb[i] * variances[i];
            }
        }

        @Override
        public double getNextStepVariance() {
            return nextStepVariance;
        }

        @Override
        public double calcVaR(double p) {
            double quantile = NID.inverseCumulativeProbability(1 - p);
            return quantile * Math.sqrt(nextStepVariance);
        }
    }

    private class MaxLikelihoodGARCHFunction implements MultivariateFunction {

        private final double uncVariance;
        private final double[] squaredValues;

        public MaxLikelihoodGARCHFunction(double[] series) {
            this.uncVariance = VARIANCE.evaluate(series);
            this.squaredValues = DoubleStream.of(series).map(v -> Math.pow(v, 2.)).toArray();
        }

        @Override
        public double value(double[] param) {
            double[] variances = calcVariances(param);
            double lml = 0;
            for (int i = 1; i < variances.length; i++) {
                double lmlp = Math.log(1 / Math.sqrt(variances[i] * 2 * Math.PI)) - squaredValues[i] / variances[i] / 2;
                lml += lmlp;
            }
            return lml;
        }

        public double[] calcVariances(double[] param) {
            double omega = param[0];
            double[] aa = Arrays.copyOfRange(param, 1, p + 1);
            double[] bb = Arrays.copyOfRange(param, p + 1, p + q + 1);

            int n = squaredValues.length;
            double[] variances = new double[n];

            variances[0] = uncVariance;
            for (int i = 1; i < n; i++) {
                variances[i] = omega;
                for (int j = 0; j < p; j++) {
                    int index = i - j - 1;
                    variances[i] += aa[j] * (index < 0 ? 0 : squaredValues[index]);
                }
                for (int j = 0; j < q; j++) {
                    int index = i - j - 1;
                    variances[i] += bb[j] * (index < 0 ? 0 : variances[index]);
                }
            }

            return variances;
        }

        public double[] getVariancesValues(double[] param) {
            return Arrays.copyOfRange(calcVariances(param), squaredValues.length - q, squaredValues.length);
        }

        public double[] getSquaredValues() {
            return Arrays.copyOfRange(squaredValues, squaredValues.length - p, squaredValues.length);
        }

        public double getUnconditionalVariance() {
            return uncVariance;
        }

    }
}
