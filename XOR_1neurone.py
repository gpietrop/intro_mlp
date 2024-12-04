# XOR con due neuroni
def xor_neuron(x1, x2):
    # Definiamo pesi e bias per due neuroni
    w1, w2, bias = 1, 1, -1
    return int((x1 * w1 + x2 * w2 + bias) > 0)


# Testiamo la funzione
inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
for x1, x2 in inputs:
    print(f"Input: ({x1}, {x2}), Output: {xor_neuron(x1, x2)}")
