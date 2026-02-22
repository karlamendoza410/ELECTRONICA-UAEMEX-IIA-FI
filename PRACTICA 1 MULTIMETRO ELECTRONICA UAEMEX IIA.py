import numpy as np
import matplotlib.pyplot as plt

# =========================
# 1. VALORES DE REFERENCIA
# =========================
# Valores nominales en MΩ
ref_A = np.array([0.001]*10 + [0.01]*10 + [0.1]*10)

ref_B = np.array([0.001]*10 + [0.01]*10 + [0.1]*10 + [1]*10)

ref_C = np.array([0.001]*10 + [0.01]*10 + [0.1]*10 +
                 [1]*10 + [10]*10)

# =========================
# 2. DATOS DE MEDICIÓN
# =========================

# PRUEBA A (AEMC)
d_a = np.concatenate([
    np.array([0.997, 0.989, 0.985, 0.999, 0.985, 0.986, 0.981, 0.984, 0.990, 0.985]) / 1000,
    np.array([9.75, 9.80, 9.76, 9.75, 9.76, 9.77, 9.80, 9.77, 9.80, 9.71]) / 1000,
    np.array([98.15, 98.61, 98.37, 98.58, 98.59, 98.84, 98.31, 98.66, 98.57, 98.28]) / 1000
])

# PRUEBA B (TRUPER)
d_b = np.concatenate([
    np.array([0.000997, 0.000984, 0.000996, 0.000989, 0.000981, 0.000983, 0.000985, 0.000983, 0.000987, 0.000985]),
    np.array([0.00969, 0.00975, 0.00972, 0.00976, 0.00977, 0.00974, 0.00974, 0.00973, 0.00972, 0.00971]),
    np.array([0.096, 0.097, 0.097, 0.096, 0.096, 0.096, 0.096, 0.097, 0.097, 0.096]),
    np.array([1.003, 0.992, 0.994, 0.97, 0.973, 1.013, 0.999, 1, 0.972, 0.995])
])

# PRUEBA C (AGILENT 34401A)
d_c = np.concatenate([
    np.array([1.004, 0.99577, 0.99038, 0.99435, 1.0023, 0.98979, 0.99027, 0.98987, 0.98989, 1.0023]) / 1000,
    np.array([9.8031, 9.8034, 9.8239, 9.7958, 9.7579, 9.8047, 9.7992, 9.8464, 9.8354, 9.8076]) / 1000,
    np.array([98.35, 98.35, 98.779, 98.792, 98.484, 98.523, 98.766, 98.668, 98.837, 98.549]) / 1000,
    np.array([1.00054, 1.01277, 0.98016, 0.97928, 0.98420, 1.00108, 1.0063, 1.0047, 0.98095, 1.0023]),
    np.array([10.266, 10.289, 10.390, 10.317, 10.338, 10.332, 10.354, 10.886, 10.226, 10.500])
])

# =========================
# FUNCIÓN PARA GRAFICAR
# =========================

def analizar_prueba(ref, data, titulo):

    error_pct = (data - ref) / ref * 100

    # ---- Gráfica 1: Linealidad ----
    plt.figure()
    plt.scatter(ref, data)
    plt.plot([min(ref), max(ref)], [min(ref), max(ref)])
    plt.xlabel("Valor de Referencia (MΩ)")
    plt.ylabel("Valor Medido (MΩ)")
    plt.title(f"{titulo} - Valor Medido vs Referencia")
    plt.show()

    # ---- Gráfica 2: Distribución Error % ----
    plt.figure()
    plt.hist(error_pct, bins=10)
    plt.xlabel("Error porcentual (%)")
    plt.ylabel("Frecuencia")
    plt.title(f"{titulo} - Distribución de Error Porcentual")
    plt.show()

    # Métricas útiles
    print(f"\n{titulo}")
    print("Error promedio (%):", np.mean(error_pct))
    print("Desviación estándar error (%):", np.std(error_pct))


# =========================
# EJECUCIÓN
# =========================

analizar_prueba(ref_A, d_a, "Prueba A (AEMC)")
analizar_prueba(ref_B, d_b, "Prueba B (Truper)")
analizar_prueba(ref_C, d_c, "Prueba C (Agilent 34401A)")


import numpy as np

# =========================
# DATOS YA DEFINIDOS
# (usa tus arreglos d_a, d_b y d_c)
# =========================

# ---- Promedios por bloque (mismo valor nominal) ----
def promedios_por_bloque(data, bloques):
    return np.array([np.mean(data[i:i+10]) for i in range(0, bloques*10, 10)])

# PRUEBA A (3 bloques: 1k, 10k, 100k)
mean_A = promedios_por_bloque(d_a, 3)

# PRUEBA B (4 bloques: 1k, 10k, 100k, 1M)
mean_B = promedios_por_bloque(d_b, 4)

# PRUEBA C (profesor)
# Tomamos solo los bloques equivalentes para comparar
mean_C_A = promedios_por_bloque(d_c[:30], 3)   # para comparar con A
mean_C_B = promedios_por_bloque(d_c[:40], 4)   # para comparar con B

# =========================
# ERROR RELATIVO %
# =========================

error_A = (mean_A - mean_C_A) / mean_C_A * 100
error_B = (mean_B - mean_C_B) / mean_C_B * 100

# =========================
# RESULTADOS
# =========================

print("Error relativo porcentual respecto al multímetro del profesor\n")

print("Prueba A (AEMC):")
for i, err in enumerate(error_A):
    print(f"Bloque {i+1}: {err:.6f} %")

print("\nPrueba B (Truper):")
for i, err in enumerate(error_B):
    print(f"Bloque {i+1}: {err:.6f} %")

# Error promedio global
print("\nError promedio global A:", np.mean(error_A), "%")
print("Error promedio global B:", np.mean(error_B), "%")


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# =========================
# FUNCIONES AUXILIARES
# =========================

def promedios_por_bloque(data, bloques):
    return np.array([np.mean(data[i:i+10]) for i in range(0, bloques*10, 10)])

def error_porcentual(medido, referencia):
    return (medido - referencia) / referencia * 100


# =========================
# PROMEDIOS POR BLOQUE
# =========================

mean_A = promedios_por_bloque(d_a, 3)
mean_B = promedios_por_bloque(d_b, 4)
mean_C_A = promedios_por_bloque(d_c[:30], 3)
mean_C_B = promedios_por_bloque(d_c[:40], 4)

# Valores nominales por bloque
ref_vals_A = np.array([0.001, 0.01, 0.1])
ref_vals_B = np.array([0.001, 0.01, 0.1, 1])
ref_vals_C = np.array([0.001, 0.01, 0.1, 1, 10])

# =========================
# TABLA COMPARATIVA A vs B
# =========================

error_A = error_porcentual(mean_A, mean_C_A)
error_B = error_porcentual(mean_B, mean_C_B)

tabla = pd.DataFrame({
    "Valor Referencia (MΩ)": ref_vals_B,
    "Promedio A (MΩ)": np.append(mean_A, np.nan),
    "Error A vs Agilent (%)": np.append(error_A, np.nan),
    "Promedio B (MΩ)": mean_B,
    "Error B vs Agilent (%)": error_B
})

print("\nTABLA COMPARATIVA PRÁCTICA A vs B\n")
print(tabla.round(6))


# =========================
# 1️⃣ ANÁLISIS DE RANGO (UNA SOLA GRÁFICA)
# =========================

mean_C_full = promedios_por_bloque(d_c, 5)

plt.figure(figsize=(8,6))
plt.plot(ref_vals_A, mean_A, 'o-', label="AEMC")
plt.plot(ref_vals_B, mean_B, 's-', label="Truper")
plt.plot(ref_vals_C, mean_C_full, '^-', label="Agilent 34401A")

plt.xlabel("Valor de Referencia (MΩ)")
plt.ylabel("Valor Medido Promedio (MΩ)")
plt.title("Análisis de Rango Comparativo")
plt.legend()
plt.grid()
plt.show()


# =========================
# IDENTIFICAR PÉRDIDA DE PRECISIÓN (>1%)
# =========================

print("\nANÁLISIS DE LÍMITE DE PRECISIÓN (>1%)\n")

for i, err in enumerate(error_A):
    if abs(err) > 1:
        print(f"AEMC pierde precisión en {ref_vals_A[i]} MΩ")

for i, err in enumerate(error_B):
    if abs(err) > 1:
        print(f"Truper pierde precisión en {ref_vals_B[i]} MΩ")


# =========================
# 2️⃣ GRÁFICA DE LINEALIDAD CONJUNTA
# =========================

plt.figure(figsize=(8,6))

plt.scatter(ref_A, d_a, alpha=0.6, label="AEMC")
plt.scatter(ref_B, d_b, alpha=0.6, label="Truper")
plt.scatter(ref_C, d_c, alpha=0.4, label="Agilent 34401A")

plt.plot([0.001, 10], [0.001, 10], 'k--', label="Ideal")

plt.xlabel("Valor de Referencia (MΩ)")
plt.ylabel("Valor Medido (MΩ)")
plt.title("Linealidad Comparativa de los Tres Equipos")
plt.legend()
plt.grid()
plt.show()


import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# --- 1. PREPARACIÓN DE DATOS ---

# PRUEBA A (AEMC)
d_a = np.concatenate([
    np.array([0.997, 0.989, 0.985, 0.999, 0.985, 0.986, 0.981, 0.984, 0.990, 0.985]) / 1000,
    np.array([9.75, 9.80, 9.76, 9.75, 9.76, 9.77, 9.80, 9.77, 9.80, 9.71]) / 1000,
    np.array([98.15, 98.61, 98.37, 98.58, 98.59, 98.84, 98.31, 98.66, 98.57, 98.28]) / 1000
])
residuos_a = d_a - np.mean(d_a)

# PRUEBA B (TRUPER)
d_b = np.concatenate([
    np.array([0.000997, 0.000984, 0.000996, 0.000989, 0.000981, 0.000983, 0.000985, 0.000983, 0.000987, 0.000985]),
    np.array([0.00969, 0.00975, 0.00972, 0.00976, 0.00977, 0.00974, 0.00974, 0.00973, 0.00972, 0.00971]),
    np.array([0.096, 0.097, 0.097, 0.096, 0.096, 0.096, 0.096, 0.097, 0.097, 0.096]),
    np.array([1.003, 0.992, 0.994, 0.97, 0.973, 1.013, 0.999, 1, 0.972, 0.995])
])
residuos_b = d_b - np.mean(d_b)

# PRUEBA C (AGILENT 34401A) - Datos de tu tabla
d_c = np.concatenate([
    np.array([1.004, 0.99577, 0.99038, 0.99435, 1.0023, 0.98979, 0.99027, 0.98987, 0.98989, 1.0023]) / 1000,
    np.array([9.8031, 9.8034, 9.8239, 9.7958, 9.7579, 9.8047, 9.7992, 9.8464, 9.8354, 9.8076]) / 1000,
    np.array([98.35, 98.35, 98.779, 98.792, 98.484, 98.523, 98.766, 98.668, 98.837, 98.549]) / 1000,
    np.array([1.00054, 1.01277, 0.98016, 0.97928, 0.98420, 1.00108, 1.0063, 1.0047, 0.98095, 1.0023]),
    np.array([10.266, 10.289, 10.390, 10.317, 10.338, 10.332, 10.354, 10.886, 10.226, 10.500])
])
residuos_c = d_c - np.mean(d_c)


# --- 2. FUNCIONES DE GRAFICACIÓN ---

def graficar_individual(residuos, titulo, color):
    plt.figure(figsize=(8, 5))
    mu, sigma = np.mean(residuos), np.std(residuos)
    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 1000)
    plt.plot(x, stats.norm.pdf(x, mu, sigma), color=color, lw=3, label=f'Error Real ($\sigma$={sigma:.6f})')
    plt.fill_between(x, stats.norm.pdf(x, mu, sigma), color=color, alpha=0.2)
    plt.axvline(0, color='black', linestyle='--', label='Valor Nominal')
    plt.title(f'Distribución de Error: {titulo}')
    plt.legend()
    plt.grid(alpha=0.3)


def graficar_ideal():
    plt.figure(figsize=(8, 5))
    # La gráfica perfecta es una línea vertical en 0 (Sigma = 0)
    plt.axvline(0, color='green', lw=5, label='Gráfica Perfecta (Sigma = 0.000000)')
    plt.xlim(-0.1, 0.1)
    plt.ylim(0, 10)
    plt.title('Modelo de Medición Perfecta (Sin Ruido Térmico)')
    plt.xlabel('Desviación (MΩ)')
    plt.ylabel('Certeza Absoluta')
    plt.legend()
    plt.grid(alpha=0.3)


def graficar_unificada():
    plt.figure(figsize=(10, 6))
    datasets = [(residuos_a, 'red', 'Prueba A'), (residuos_b, 'magenta', 'Prueba B'), (residuos_c, 'cyan', 'Prueba C')]
    for data, color, label in datasets:
        mu, sigma = np.mean(data), np.std(data)
        x = np.linspace(-0.3, 0.3, 2000)
        plt.plot(x, stats.norm.pdf(x, mu, sigma), color=color, lw=2, label=f'{label} ($\sigma$={sigma:.5f})')

    # Agregar la línea ideal para contraste
    plt.axvline(0, color='green', lw=2, linestyle='-', label='IDEAL (Resistencia Nominal)')

    plt.title('Comparativa de Instrumentos vs. Modelo Ideal')
    plt.xlabel('Error de Medición (MΩ)')
    plt.ylabel('Densidad de Probabilidad')
    plt.legend()
    plt.grid(alpha=0.3)


# --- 3. EJECUCIÓN ---

graficar_individual(residuos_a, "AEMC", "red")
graficar_individual(residuos_b, "Truper", "magenta")
graficar_individual(residuos_c, "Agilent 34401A", "cyan")
graficar_ideal()  # Gráfica Perfecta
graficar_unificada()  # Comparativa con Ideal

plt.show()