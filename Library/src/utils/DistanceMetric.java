package utils;

import java.util.List;

public interface DistanceMetric<F> {
    double calculate(List<F> f1, List<F> f2);
}