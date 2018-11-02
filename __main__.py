from rayt import *
from scenes import *

with torch.no_grad():
    #render(cornell())
    render(cyl())
