package Evaluation;

import DataProcessing.domain.Instance;

import java.util.List;

public class Accuracy<F, L> implements EvaluationMeasure<F, L> {

    @Override
    public double evaluate(List<Instance<F, L>> instances, List<L> predictions) {
        if (instances.size() != predictions.size()) {
            throw new IllegalArgumentException("Number of instances and predictions must match");
        }

        long correct = 0;
        for (int i = 0; i < instances.size(); i++) {
            if (instances.get(i).getOutput().equals(predictions.get(i))) {
                correct++;
            }
        }
        return (double) correct / instances.size();
    }
}

