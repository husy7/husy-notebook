prices = [7,1,5,3,6,4]
max = 0
low = min(prices)
if low != prices[-1]:
    for i in prices[prices.index(low):]:
        if max <= i:
            max = i
mo = max - low
if mo <0:
    mo = 0
print(mo)


