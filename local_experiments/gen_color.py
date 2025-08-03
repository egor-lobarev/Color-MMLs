import random
from typing import List, Tuple, Dict


class MunsellColor:
    def __init__(self, h: float, c: int, v: int):
        self.h = h  # оттенок
        self.c = c  # насыщенность
        self.v = v  # яркость

    def __repr__(self):
        return f"H={self.h}, C={self.c}, V={self.v}"
    
    @staticmethod
    def munsell_color_pairs(n_pairs: int, chain_type: str='hue'):
        """
        Генерирует пары цветов Munssel с изменением одного параметра (hue, chroma, value).
        
        :param n_pairs: количество генерируемых пар
        :param chain_type: какой параметр изменять ('hue', 'chroma', 'value')
        :return: список пар цветов
        """
        if chain_type == 'hue':
            hue_values = list(range(0, 10)) + ['2.5Y']
            colors = [(MunsellColor(h=h_val, c=random.randint(1, 10), v=random.randint(1, 10)), 
                      MunsellColor(h=(float(h_val)+2.5)%10, c=random.randint(1, 10), v=random.randint(1, 10))) 
                     for _ in range(n_pairs)]
            
        elif chain_type == 'chroma':
            colors = [(MunsellColor(h=random.uniform(0, 10), c=c_val, v=random.randint(1, 10)),
                      MunsellColor(h=random.uniform(0, 10), c=c_val+1, v=random.randint(1, 10)))
                     for c_val in range(random.randint(1, 9))]
            
        else:  # Изменение яркости
            colors = [(MunsellColor(h=random.uniform(0, 10), c=random.randint(1, 10), v=v_val),
                      MunsellColor(h=random.uniform(0, 10), c=random.randint(1, 10), v=v_val+1))
                     for v_val in range(random.randint(1, 9))]
            
        return colors[:n_pairs]

def generate_control_pairs(colors: List[MunsellColor], distance: float=2.5) -> List[Tuple[MunsellColor, MunsellColor]]:
    """Генерируем контрольные пары с большим воспринимаемым различием."""
    control_pairs = []
    for col in colors:
        # Случайно выбираем пару цветов из другого диапазона оттенков
        other_hue = random.choice([val for val in range(0, 10) if abs(val-col.h) > distance])
        new_col = MunsellColor(h=other_hue, c=col.c, v=col.v)
        control_pairs.append((col, new_col))
    return control_pairs
  
