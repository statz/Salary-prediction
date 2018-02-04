import graphviz as gv
import os
from graphviz import Source
text = ''.join(list(((open("2").read().splitlines()))))
g2 = Source(str(text))

g2.render('g2.gv', view=True)