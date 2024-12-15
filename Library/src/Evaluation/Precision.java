package Evaluation;

import DataProcessing.domain.Instance;

import java.util.List;

public class Precision<F, L> implements EvaluationMeasure<F, L> {
    @Override
    public double evaluate(List<Instance<F, L>> instances, List<L> predictions) {
        if (instances.size() != predictions.size()) {
            throw new IllegalArgumentException("Number of instances and predictions must match");
        }

        long truePositives = 0;
        long falsePositives = 0;

        for (int i = 0; i < instances.size(); i++) {
            L predicted = predictions.get(i);
            L actual = instances.get(i).getOutput();

            if (predicted.equals(1)) {
                if (actual.equals(1)) {
                    truePositives++;
                } else {
                    falsePositives++;
                }
            }
        }

        // Avoid division by zero
        if (truePositives + falsePositives == 0) {
            return 0.0;
        }

        return (double) truePositives / (truePositives + falsePositives);
    }
}