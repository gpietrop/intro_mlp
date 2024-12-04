# Un singolo neurone
def simple_neuron(input_value, weight, bias):
    return input_value * weight + bias


# Prova il neurone
input_value = 2
weight = 0.5
bias = 1
output = simple_neuron(input_value, weight, bias)
print("Output del neurone:", output)
