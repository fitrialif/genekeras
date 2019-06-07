# GeneKeras

## What is GeneKeras
...........

## Installation
Install using pip:
```
pip install git+https://github.com/d-corsi/genekeras.git
```
Update using pip:
```
pip install --upgrade git+https://github.com/d-corsi/genekeras.git
```
Remove using pip:
```
pip uninstall git+https://github.com/d-corsi/genekeras.git
```

# Example

```
import genekeras

gk = genekeras.GeneKeras(load_compiled = True)
gk.set_parents(mom, dad)
gk.set_param(crossover_enabled = False, mutation_enabled = False, mutation_prob = 0.1, mutation_rate = 0.5)
child = gk.get_child()
```

# TODOS