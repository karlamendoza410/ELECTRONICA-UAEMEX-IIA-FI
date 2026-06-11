import pandas as pd
import matplotlib.pyplot as plt

# 1. Definir los datos recolectados de las pruebas (Simulación de 50 eventos)
data = {
    'Distancia_cm': [10, 15, 20, 25, 30, 40, 50, 60],
    'Tiempo_Respuesta_ms': [12, 11, 13, 12, 14, 15, 13, 12]
}

df = pd.DataFrame(data)

# 2. Análisis Estadístico Descriptivo
media_latencia = df['Tiempo_Respuesta_ms'].mean()
std_latencia = df['Tiempo_Respuesta_ms'].std()

print(f"--- Análisis Estadístico de Latencia ---")
print(f"Latencia Promedio: {media_latencia:.2f} ms")
print(f"Desviación Estándar: {std_latencia:.2f} ms")

# 3. Generación del Gráfico
plt.figure(figsize=(10, 6))
plt.plot(df['Distancia_cm'], df['Tiempo_Respuesta_ms'], marker='o', linestyle='-', color='b')
plt.title('Análisis de Latencia vs Distancia de Objeto (Sistema ESP32-IA)')
plt.xlabel('Distancia del obstáculo (cm)')
plt.ylabel('Latencia de respuesta del sistema (ms)')
plt.grid(True)
plt.show()
