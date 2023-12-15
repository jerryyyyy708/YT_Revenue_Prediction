import pandas as pd

def get_class(revenue):
    labels=[]
    for i in revenue:
        if (i>=3000):
            labels.append(3)
        elif(i>=2000):
            labels.append(2)
        elif(i>=1000):
            labels.append(1)
        else:
            labels.append(0)
    return labels
    