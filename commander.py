import math
from settings import settings

def calc_throttle(speed, turn_rate=0, speed_limit=settings["TARGET_SPEED"]):
    throttle = (1 / (1 + math.e ** (speed - speed_limit)) - 0.5) * 2 * (1 - abs(turn_rate))
    return throttle

"""
Spiegazione formula:
Parto dalla funzione logistica (sigmoide) 1/1+e^x -> R[+1,0]
Porto questa funzione nel raggio di possibili valori R[+0.5,-0.5] sottraendo 0.5
La porta nel raggio R[+1,-1] moltiplicando il tutto per 2
Al momento la funzione vale 0 quando x vale x, per fare in modo che la funzione valga 0
quando x vale la velocità desiderata, sottraggo a x la velocità desiderata
Per adattare la velocità allo sterzo:
Possibili valori dello sterzo   -1 ----- 0 ----- 1
Corrispettivi valori del coeff.  0 ----- 1 ----- 0
Per farlo sottraggo 1 al valore assoluto dello sterzo
"""
