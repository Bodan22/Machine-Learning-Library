package utils;

import java.util.List;

public class DiabetesDistanceMetric implements DistanceMetric<Double> {
    private final double[] minValues;
    private final double[] maxValues;
    private final double[] weights;

    public DiabetesDistanceMetric(double[] minValues, double[] maxValues) {
        if (minValues.length != 8 || maxValues.length != 8) {
            throw new IllegalArgumentException("Min and max values arrays must have length 8");
        }
        this.minValues = minValues.clone();
        this.maxValues = maxValues.clone();

        this.weights = new double[]{
                0.5,  // Pregnancies
                1.0,  // Glucose
                0.8,  // BloodPressure
                0.7,  // SkinThickness
                0.6,  // Insulin
                1.0,  // BMI
                0.9,  // DiabetesPedigreeFunction
                0.7   // Age
        };
    }

    @Override
    public double calculate(List<Double> f1, List<Double> f2) {
        if (f1 == null || f2 == null || f1.size() != 8 || f2.size() != 8) {
            throw new IllegalArgumentException("Feature vectors must have length 8");
        }

        double sumSquaredDiff = 0.0;
        for (int i = 0; i < 8; i++) {
            if (isValidFeature(i, f1.get(i)) && isValidFeature(i, f2.get(i))) {
                double norm1 = normalize(f1.get(i), i);
                double norm2 = normalize(f2.get(i), i);
                sumSquaredDiff += weights[i] * Math.pow(norm1 - norm2, 2);
            }
        }

        return Math.sqrt(sumSquaredDiff);
    }

    private double normalize(double value, int featureIndex) {
        return (value - minValues[featureIndex]) /
                (maxValues[featureIndex] - minValues[featureIndex]);
    }

    private boolean isValidFeature(int index, double value) {
        switch (index) {
            case 0: // Pregnancies can be 0
                return true;
            case 1: // Glucose
            case 2: // BloodPressure
            case 3: // SkinThickness
                return value > 0;
            case 4: // Insulin can be 0
                return true;
            case 5: // BMI
                return value > 0;
            case 6: // DiabetesPedigreeFunction
                return value >= 0;
            case 7: // Age
                return value > 0;
            default:
                return false;
        }
    }
}