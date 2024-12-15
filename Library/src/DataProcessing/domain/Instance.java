package DataProcessing.domain;

import java.util.List;

public class Instance<F, L> {
    private final List<F> input;
    private final L output;

    public Instance(List<F> input, L output) {
        this.input = input;
        this.output = output;
    }

    public List<F> getInput() {
        return input;
    }

    public L getOutput() {
        return output;
    }
}