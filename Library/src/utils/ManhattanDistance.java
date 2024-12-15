package utils;

import java.util.List;

public class ManhattanDistance implements DistanceMetric<Double> {

    @Override
    public double calculate(List<Double> a, List<Double> b) {
        if (a == null || b == null) {
            throw new IllegalArgumentException("Input lists cannot be null");
        }

        if (a.size() != b.size()) {
            throw new IllegalArgumentException("Input lists must have the same size. Found: "
                    + a.size() + " and " + b.size());
        }

        double sumOfAbsoluteDifferences = 0.0;

        for (int i = 0; i < a.size(); i++) {
            Double val1 = a.get(i);
            Double val2 = b.get(i);

            // Handle null values
            if (val1 == null || val2 == null) {
                throw new IllegalArgumentException("Values cannot be null at index " + i);
            }

            sumOfAbsoluteDifferences += Math.abs(val1 - val2);
        }

        return sumOfAbsoluteDifferences;
    }
}