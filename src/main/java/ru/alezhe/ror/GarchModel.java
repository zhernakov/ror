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
import org.apache.commons.math3.optim.linear.LinearConstraint;
import org.apache.commons.math3.optim.linear.LinearConstraintSet;
import org.apache.commons.math3.optim.linear.Relationship;
import org.apache.commons.math3.optim.nonlinear.scalar.GoalType;
import org.apache.commons.math3.optim.nonlinear.scalar.MultivariateOptimizer;
import org.apache.commons.math3.optim.nonlinear.scalar.ObjectiveFunction;
import org.apache.commons.math3.optim.nonlinear.scalar.noderiv.BOBYQAOptimizer;
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

    //////////////////////////
    public GarchApproximatedTimeSeries approximateTimeSeries(ObservedTimeSeries ts, int windowSize) {
        GarchApproximatedTimeSeries apprTs = new GarchApproximatedTimeSeries(ts, windowSize);
        apprTs.approximate();
        return apprTs;
    }

    public MarGarchApproximatedTimeSeries approximateMarTimeSeries(ObservedTimeSeries ts, int windowSize, double confidence) {
        MarGarchApproximatedTimeSeries apprTs = new MarGarchApproximatedTimeSeries(ts, windowSize, confidence);
        apprTs.approximate();
        return apprTs;
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

        protected final GarchApproximation[] approximations;
        protected final double[][] variances;
        private final int windowSize;

        protected GarchApproximatedTimeSeries(ObservedTimeSeries series, int windowSize) {
            super(series);
            this.approximations = new GarchApproximation[n];
            this.variances = new double[n][];
            this.variances[0] = new double[q];
            Arrays.fill(variances[0], VARIANCE.evaluate(values));
            this.windowSize = windowSize;
        }

        protected void approximate() {
            for (int i = 1; i < n; i++) {
                approximateAtTimePoint(i);
            }
        }

        final 
        protected void approximateAtTimePoint(int timePoint) {
            GarchApproximation aprx = fitGarchParameters(timePoint);
            if (aprx == null) {
                aprx = approximations[timePoint - 1];
                System.err.println("failed to optimize at timepoint " + timePoint);
            }
            System.out.println(timePoint);
            approximations[timePoint] = aprx;
            if (aprx != null) {
                double[] fitVars = calcVariances(aprx, timePoint);
                variances[timePoint] = Arrays.copyOfRange(fitVars, fitVars.length - q, fitVars.length);
            } else {
                variances[timePoint] = variances[timePoint - 1];
            }
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
                return null;
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
                return lml;
            }
        }
    }

    public class MarGarchApproximatedTimeSeries extends GarchApproximatedTimeSeries {
        private final double confidence;
        private final GarchApproximation[] marApproximations;
        private final double[][] marVariances;
        
        private final Map<double[],double[]> failsHistory = new LinkedHashMap<>();
        
        private MarGarchApproximatedTimeSeries(ObservedTimeSeries series, int windowSize, double confidence) {
            super(series, windowSize);
            this.confidence = confidence;
            this.marApproximations = new GarchApproximation[n + 1];
            this.marVariances = new double[n + 1][];
            
            this.marVariances[0] = variances[0];
        }

        @Override
        protected void approximate() {
            for (int i = 1; i < n; i++) {
                approximateAtTimePoint(i);
                
                double var = getVaR(i, confidence);
                boolean failed = values[i] < var;
                if (failed) {
                    if (i == 1) {
                        throw new UnsupportedOperationException("we have not been ready to first fall!!!");
                    }
                    addToFailsHistory(i);
                    GarchApproximation apr = calculateApproximation();
                    if (apr == null) {
                        apr = marApproximations[i];
                        System.err.println("failed to optimize MaR at timepoint " + i);
                    }
                    marApproximations[i + 1] = apr;
                } else {
                    if (marApproximations[i] == null) {
                        marApproximations[i] = approximations[i];
                        marVariances[i] = variances[i];
                    }
                    marApproximations[i + 1] = marApproximations[i];
                }
                marVariances[i + 1] = calcMarVariances(marApproximations[i + 1], i + 1);
            }
        }

        private void addToFailsHistory(int timePoint) {
            double[] vals = Arrays.copyOfRange(values, timePoint - p + 1, timePoint + 1);
            double[] vars = marVariances[timePoint];
            
            failsHistory.put(vals, vars);
        }

        protected double[] calcMarVariances(GarchApproximation apr, int timePoint) {
            final int realSize = Math.min(q, timePoint);
            final int first = timePoint - realSize;
            final double[] vars = new double[realSize + q];
            System.arraycopy(marVariances[first], 0, vars, 0, q);
            for (int i = 0; i < realSize; i++) {
                vars[i + q] = apr.getVariance(first + i, values, i + q, vars);
            }
            return Arrays.copyOfRange(vars, realSize, realSize + q);
        }
        
        private GarchApproximation calculateApproximation() {
            MultivariateOptimizer optimizer = new BOBYQAOptimizer((p + q + 1) * 1 + (2));
            MarGarchMaxLikelihoodFunction lmfun = new MarGarchMaxLikelihoodFunction();
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
                return null;
            }
        }

        double getMaR(int timePoint) {
            GarchApproximation apr = marApproximations[timePoint];
            double variance = apr.getVariance(timePoint, values, q, marVariances[timePoint]);
            
            double quantile = NID.inverseCumulativeProbability(1 - confidence);
            return quantile * Math.sqrt(variance);
        }

        private class MarGarchMaxLikelihoodFunction implements MultivariateFunction {

            @Override
            public double value(double[] params) {
                final GarchApproximation apr = new GarchApproximation(params);
                
                double lml = 0;
                for (Map.Entry<double[], double[]> entry : failsHistory.entrySet()) {
                    double[] vals = entry.getKey();
                    double[] vars = entry.getValue();
                    double variance = apr.getVariance(p, vals, q, vars);
                    double lmlp = Math.log(1 / Math.sqrt(variance * 2 * Math.PI)) - Math.pow(vals[p - 1], 2) / variance / 2;
                    lml += lmlp;
                }
                return lml;
            }
            
        }
        
    }
    //////////////////////////
    //////////////////////////
}
