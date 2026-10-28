

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.Random;

public final class QuantumCore {
    private QuantumCore() {}

    // Complex numbers are represented as immutable values.
    public record Complex(double re, double im) {
        public static final Complex ZERO = new Complex(0, 0);
        public static final Complex ONE = new Complex(1, 0);

        public Complex {
            if (!Double.isFinite(re) || !Double.isFinite(im)) {
                throw new IllegalArgumentException("Complex components must be finite.");
            }
        }

        public Complex add(Complex other) {
            return new Complex(re + other.re, im + other.im);
        }

        public Complex multiply(Complex other) {
            return new Complex(
                re * other.re - im * other.im,
                re * other.im + im * other.re
            );
        }

        public Complex scale(double factor) {
            return new Complex(re * factor, im * factor);
        }

        public double absSquared() {
            return re * re + im * im;
        }

        public Complex conjugate() {
            return new Complex(re, -im);
        }

        @Override
        public String toString() {
            return String.format("%.5f%+.5fi", re, im);
        }
    }

    /*
     * A 2x2 complex matrix. Entries are copied on construction so callers
     * cannot mutate a gate after it has been added to a circuit.
     */
    public static final class Gate2 {
        private final Complex[][] m;

        public Gate2(Complex a00, Complex a01, Complex a10, Complex a11) {
            m = new Complex[][] {
                {Objects.requireNonNull(a00), Objects.requireNonNull(a01)},
                {Objects.requireNonNull(a10), Objects.requireNonNull(a11)}
            };

            if (!isUnitary(1e-10)) {
                throw new IllegalArgumentException("Gate matrix must be unitary.");
            }
        }

        public Complex get(int row, int col) {
            return m[row][col];
        }

        public boolean isUnitary(double tolerance) {
            // Check U†U = I, using the two columns of U.
            for (int c1 = 0; c1 < 2; c1++) {
                for (int c2 = 0; c2 < 2; c2++) {
                    double re = 0;
                    double im = 0;

                    for (int r = 0; r < 2; r++) {
                        Complex left = m[r][c1].conjugate();
                        Complex right = m[r][c2];
                        Complex product = left.multiply(right);

                        re += product.re();
                        im += product.im();
                    }

                    double expected = c1 == c2 ? 1.0 : 0.0;

                    if (Math.abs(re - expected) > tolerance
                            || Math.abs(im) > tolerance) {
                        return false;
                    }
                }
            }

            return true;
        }
    }

    public static final class Gates {
        private Gates() {}

        public static final Gate2 I =
            gate(Complex.ONE, Complex.ZERO, Complex.ZERO, Complex.ONE);

        public static final Gate2 X =
            gate(Complex.ZERO, Complex.ONE, Complex.ONE, Complex.ZERO);

        public static final Gate2 Y = gate(
            Complex.ZERO, new Complex(0, -1),
            new Complex(0, 1), Complex.ZERO
        );

        public static final Gate2 Z =
            gate(Complex.ONE, Complex.ZERO, Complex.ZERO, new Complex(-1, 0));

        public static Gate2 H() {
            double s = 1.0 / Math.sqrt(2.0);

            return gate(
                new Complex(s, 0), new Complex(s, 0),
                new Complex(s, 0), new Complex(-s, 0)
            );
        }

        public static Gate2 phase(double theta) {
            return gate(
                Complex.ONE,
                Complex.ZERO,
                Complex.ZERO,
                new Complex(Math.cos(theta), Math.sin(theta))
            );
        }

        public static Gate2 rx(double theta) {
            double c = Math.cos(theta / 2.0);
            double s = Math.sin(theta / 2.0);

            return gate(
                new Complex(c, 0), new Complex(0, -s),
                new Complex(0, -s), new Complex(c, 0)
            );
        }

        // Conventional Rz(theta) = diag(e^(-i theta/2), e^(i theta/2)).
        public static Gate2 rz(double theta) {
            return gate(
                new Complex(
                    Math.cos(theta / 2),
                    -Math.sin(theta / 2)
                ),
                Complex.ZERO,
                Complex.ZERO,
                new Complex(
                    Math.cos(theta / 2),
                    Math.sin(theta / 2)
                )
            );
        }

        private static Gate2 gate(
                Complex a, Complex b, Complex c, Complex d) {
            return new Gate2(a, b, c, d);
        }
    }

    public static final class Register {
        private final int qubits;
        private final int dimension;
        private final double[] re;
        private final double[] im;
        private final Random random;

        public Register(int qubits, long seed) {
            if (qubits < 1 || qubits > 24) {
                throw new IllegalArgumentException(
                    "Qubit count must be between 1 and 24 for this reference implementation."
                );
            }

            this.qubits = qubits;
            this.dimension = 1 << qubits;
            this.re = new double[dimension];
            this.im = new double[dimension];
            this.random = new Random(seed);

            re[0] = 1.0; // Initial state |00...0>
        }

        public int qubits() {
            return qubits;
        }

        public int dimension() {
            return dimension;
        }

        public Complex amplitude(int basisIndex) {
            checkBasisIndex(basisIndex);
            return new Complex(re[basisIndex], im[basisIndex]);
        }

        public double probability(int basisIndex) {
            checkBasisIndex(basisIndex);

            return re[basisIndex] * re[basisIndex]
                 + im[basisIndex] * im[basisIndex];
        }

        public double totalProbability() {
            double sum = 0.0;

            for (int i = 0; i < dimension; i++) {
                sum += probability(i);
            }

            return sum;
        }

        public void apply(Gate2 gate, int target) {
            Objects.requireNonNull(gate, "gate");
            checkQubit(target);

            // Each pair differs only in the target bit.
            // Every pair is visited exactly once.
            int stride = 1 << target;
            int block = stride << 1;

            for (int base = 0; base < dimension; base += block) {
                for (int offset = 0; offset < stride; offset++) {
                    int i0 = base + offset;
                    int i1 = i0 + stride;

                    double ar = re[i0];
                    double ai = im[i0];
                    double br = re[i1];
                    double bi = im[i1];

                    Complex u00 = gate.get(0, 0);
                    Complex u01 = gate.get(0, 1);
                    Complex u10 = gate.get(1, 0);
                    Complex u11 = gate.get(1, 1);

                    // New amplitude for the |0> branch: u00*a + u01*b
                    re[i0] =
                        u00.re() * ar - u00.im() * ai
                      + u01.re() * br - u01.im() * bi;

                    im[i0] =
                        u00.re() * ai + u00.im() * ar
                      + u01.re() * bi + u01.im() * br;

                    // New amplitude for the |1> branch: u10*a + u11*b
                    re[i1] =
                        u10.re() * ar - u10.im() * ai
                      + u11.re() * br - u11.im() * bi;

                    im[i1] =
                        u10.re() * ai + u10.im() * ar
                      + u11.re() * bi + u11.im() * br;
                }
            }
        }

        public void cnot(int control, int target) {
            checkQubit(control);
            checkQubit(target);

            if (control == target) {
                throw new IllegalArgumentException(
                    "Control and target qubits must differ."
                );
            }

            int controlMask = 1 << control;
            int targetMask = 1 << target;

            for (int i = 0; i < dimension; i++) {
                // Flip target only when control is 1 and target is 0.
                if ((i & controlMask) != 0 && (i & targetMask) == 0) {
                    int partner = i | targetMask;

                    swap(re, i, partner);
                    swap(im, i, partner);
                }
            }
        }

        public void normalize() {
            double normSquared = totalProbability();

            if (!(normSquared > 0.0) || !Double.isFinite(normSquared)) {
                throw new IllegalStateException(
                    "Cannot normalize a zero or invalid state."
                );
            }

            double inverseNorm = 1.0 / Math.sqrt(normSquared);

            for (int i = 0; i < dimension; i++) {
                re[i] *= inverseNorm;
                im[i] *= inverseNorm;
            }
        }

        /**
         * Samples a basis state according to its probability
         * and collapses the register to that state.
         */
        public int measure() {
            double total = totalProbability();

            if (!(total > 0.0) || !Double.isFinite(total)) {
                throw new IllegalStateException(
                    "State has invalid total probability."
                );
            }

            double draw = random.nextDouble() * total;
            double cumulative = 0.0;

            for (int i = 0; i < dimension; i++) {
                cumulative += probability(i);

                if (draw < cumulative || i == dimension - 1) {
                    Arrays.fill(re, 0.0);
                    Arrays.fill(im, 0.0);

                    re[i] = 1.0;
                    return i;
                }
            }

            throw new AssertionError("Unreachable measurement state.");
        }

        public double[] probabilitiesCopy() {
            double[] result = new double[dimension];

            for (int i = 0; i < dimension; i++) {
                result[i] = probability(i);
            }

            return result;
        }

        private void checkQubit(int q) {
            if (q < 0 || q >= qubits) {
                throw new IndexOutOfBoundsException(
                    "Qubit index out of range: " + q
                );
            }
        }

        private void checkBasisIndex(int i) {
            if (i < 0 || i >= dimension) {
                throw new IndexOutOfBoundsException(
                    "Basis index out of range: " + i
                );
            }
        }

        private static void swap(double[] values, int a, int b) {
            double tmp = values[a];
            values[a] = values[b];
            values[b] = tmp;
        }
    }

    @FunctionalInterface
    private interface Operation {
        void apply(Register register);
    }

    public static final class Circuit {
        private final int qubits;
        private final List<Operation> operations = new ArrayList<>();

        public Circuit(int qubits) {
            if (qubits < 1 || qubits > 24) {
                throw new IllegalArgumentException(
                    "Qubit count must be between 1 and 24."
                );
            }

            this.qubits = qubits;
        }

        public Circuit h(int q) {
            return add(r -> r.apply(Gates.H(), q));
        }

        public Circuit x(int q) {
            return add(r -> r.apply(Gates.X, q));
        }

        public Circuit y(int q) {
            return add(r -> r.apply(Gates.Y, q));
        }

        public Circuit z(int q) {
            return add(r -> r.apply(Gates.Z, q));
        }

        public Circuit phase(int q, double theta) {
            return add(r -> r.apply(Gates.phase(theta), q));
        }

        public Circuit rx(int q, double theta) {
            return add(r -> r.apply(Gates.rx(theta), q));
        }

        public Circuit rz(int q, double theta) {
            return add(r -> r.apply(Gates.rz(theta), q));
        }

        public Circuit cnot(int control, int target) {
            return add(r -> r.cnot(control, target));
        }

        public Register execute(long seed) {
            Register register = new Register(qubits, seed);

            for (Operation operation : operations) {
                operation.apply(register);
            }

            return register;
        }

        public int operationCount() {
            return operations.size();
        }

        private Circuit add(Operation operation) {
            operations.add(operation);
            return this;
        }
    }

    // Basic self-tests for the Hadamard and Bell-state circuits.
    private static void runSelfTests() {
        Register bell = new Circuit(2)
            .h(0)
            .cnot(0, 1)
            .execute(7);

        assertClose(
            bell.probability(0), 0.5, 1e-10, "Bell P(|00>)"
        );
        assertClose(
            bell.probability(3), 0.5, 1e-10, "Bell P(|11>)"
        );
        assertClose(
            bell.probability(1), 0.0, 1e-10, "Bell P(|01>)"
        );
        assertClose(
            bell.probability(2), 0.0, 1e-10, "Bell P(|10>)"
        );
        assertClose(
            bell.totalProbability(), 1.0, 1e-10, "Bell normalization"
        );

        Register plus = new Circuit(1)
            .h(0)
            .execute(7);

        assertClose(
            plus.probability(0), 0.5, 1e-10, "H |0> -> P(0)"
        );
        assertClose(
            plus.probability(1), 0.5, 1e-10, "H |0> -> P(1)"
        );

        System.out.println("Self-tests passed.");
    }

    private static void assertClose(
            double actual,
            double expected,
            double tolerance,
            String label) {

        if (Math.abs(actual - expected) > tolerance) {
            throw new IllegalStateException(
                label + ": expected " + expected + ", got " + actual
            );
        }
    }

    public static void main(String[] args) {
        runSelfTests();

        int qubits = 4;

        Circuit circuit = new Circuit(qubits)
            .h(0)
            .cnot(0, 1)
            .cnot(1, 2)
            .cnot(2, 3)
            .rz(3, Math.PI / 4.0);

        long start = System.nanoTime();
        Register register = circuit.execute(2026L);
        long elapsed = System.nanoTime() - start;

        System.out.println("\nQuantumCore-JVM");
        System.out.println("Qubits: " + register.qubits());
        System.out.println(
            "State-vector dimension: " + register.dimension()
        );
        System.out.println(
            "Circuit operations: " + circuit.operationCount()
        );
        System.out.printf(
            "Total probability: %.12f%n",
            register.totalProbability()
        );
        System.out.printf(
            "Execution time: %.3f ms%n",
            elapsed / 1_000_000.0
        );

        System.out.println("Non-zero basis probabilities:");

        double[] probabilities = register.probabilitiesCopy();

        for (int i = 0; i < probabilities.length; i++) {
            if (probabilities[i] > 1e-12) {
                String bits = String.format(
                    "%" + qubits + "s",
                    Integer.toBinaryString(i)
                ).replace(' ', '0');

                System.out.printf(
                    "  |%s> : %.6f%n",
                    bits,
                    probabilities[i]
                );
            }
        }

        int measured = register.measure();

        String measuredBits = String.format(
            "%" + qubits + "s",
            Integer.toBinaryString(measured)
        ).replace(' ', '0');

        System.out.printf(
            "One sampled measurement: |%s> (index %d)%n",
            measuredBits,
            measured
        );
    }
}
