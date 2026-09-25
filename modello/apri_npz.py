import numpy as np

data = np.load("single_sample_600x6.npz")

print("Chiavi presenti nel file:")
print(data.files)

for key in data.files:
    x = data[key]

    print("\nNome array:", key)
    print("Shape:", x.shape)
    print("Tipo dati:", x.dtype)
    print("Valore minimo:", x.min())
    print("Valore massimo:", x.max())

    print("\nPrimi valori:")
    print(x)
