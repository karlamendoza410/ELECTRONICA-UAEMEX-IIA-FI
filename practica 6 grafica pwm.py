import matplotlib.pyplot as plt
import numpy as np

# Configuración de los datos de la melodía (simulación de tiempo y frecuencia)
tiempos = np.linspace(0, 3, 500)  # 3 segundos de duración total
frecuencias = [261, 330, 392, 523]  # Notas: C4, E4, G4, C5
duracion_nota = 0.75

# Generación de la señal PWM (onda cuadrada)
señal_pwm = np.zeros_like(tiempos)
for i, t in enumerate(tiempos):
    nota_idx = int(t / duracion_nota)
    if nota_idx < len(frecuencias):
        # Genera una onda cuadrada basada en la frecuencia
        señal_pwm[i] = 1 if (np.sin(2 * np.pi * frecuencias[nota_idx] * t) > 0) else 0

# Crear la gráfica
plt.figure(figsize=(12, 5))
plt.plot(tiempos, señal_pwm, color='darkblue', linewidth=1.5)

# Detalles de formato técnico
plt.title('Representación Visual de la Modulación PWM en la Melodía', fontsize=14)
plt.xlabel('Tiempo (segundos)', fontsize=12)
plt.ylabel('Estado de la Señal (Digital)', fontsize=12)
plt.ylim(-0.1, 1.1)
plt.grid(True, linestyle='--', alpha=0.6)

# Etiquetas para las notas
for i, freq in enumerate(frecuencias):
    plt.text(i * duracion_nota + 0.2, 0.5, f'{freq} Hz', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()