from machine import Pin
import time

# 1. DEFINICIÓN DE PINES
sensor_izq = Pin(0, Pin.IN)
sensor_der = Pin(1, Pin.IN)

# Ordenamos del centro hacia afuera. 
# Izquierda va en reversa (4->3->2) y derecha normal (5->6->7)
pines_izq = [4, 3, 2]
pines_der = [5, 6, 7]

leds_izq = [Pin(pin, Pin.OUT) for pin in pines_izq]
leds_der = [Pin(pin, Pin.OUT) for pin in pines_der]

velocidad = 0.1 

# --- NUEVAS VARIABLES DE MEMORIA ---
# Guardan el estado: True (Prendido) o False (Apagado)
direccional_izq_activa = False
direccional_der_activa = False

# Guardan la lectura anterior para saber cuándo "pasaste" el dedo y no cuándo lo "dejaste" ahí
sensor_izq_ant = 1
sensor_der_ant = 1

# Asegurarnos de que todos los LEDs inicien apagados
for led in leds_izq + leds_der:
    led.value(0)

# 2. FUNCIÓN DE ANIMACIÓN
def ejecutar_viborita(grupo_leds):
    # Encender en secuencia uno por uno (ahora irá del centro hacia afuera)
    for led in grupo_leds:
        led.value(1) 
        time.sleep(velocidad)
        
    # Apagar todos de golpe
    for led in grupo_leds:
        led.value(0) 
        
    time.sleep(0.2) 

# 3. BUCLE PRINCIPAL
# 3. BUCLE PRINCIPAL CON FILTRO ANTI-RUIDO
while True:
    estado_izq = sensor_izq.value()
    estado_der = sensor_der.value()
    
    # --- LÓGICA DE INTERRUPTOR CON ANTI-REBOTE ---
    # Revisamos el izquierdo
    if estado_izq == 0:
        time.sleep(0.05) # Esperamos 50ms para confirmar
        if sensor_izq.value() == 0 and sensor_izq_ant == 1: # Confirmado
            direccional_izq_activa = not direccional_izq_activa
            if direccional_izq_activa:
                direccional_der_activa = False
            time.sleep(0.3)
        sensor_izq_ant = 0 # Actualizamos estado
    else:
        sensor_izq_ant = 1
         
    # Revisamos el derecho
    if estado_der == 0:
        time.sleep(0.05) # Esperamos 50ms para confirmar
        if sensor_der.value() == 0 and sensor_der_ant == 1: # Confirmado
            direccional_der_activa = not direccional_der_activa
            if direccional_der_activa:
                direccional_izq_activa = False
            time.sleep(0.3)
        sensor_der_ant = 0 # Actualizamos estado
    else:
        sensor_der_ant = 1

    # --- EJECUCIÓN ---
    if direccional_izq_activa:
        ejecutar_viborita(leds_izq)
    elif direccional_der_activa:
        ejecutar_viborita(leds_der)
    else:
        for led in leds_izq + leds_der:
            led.value(0)
        time.sleep(0.05)