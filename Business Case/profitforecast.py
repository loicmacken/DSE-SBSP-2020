import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as ax


def cost_curve(starttime, endtime, amount):
    cost_array = np.zeros(25+6)
    timespan = endtime-starttime
    total = 0
    for i in range(starttime+6, endtime+6, 1):
        cost_array[i] = amount/timespan
        total += amount/timespan
    
    return cost_array
        

#timeline
start = -6
end =  25
time = np.arange(start,end,1)
cost_time = np.zeros(end-start)

# costs
jobs = 455
cost_jobs = cost_curve(-6, 25, 455)

infra = 500
cost_infra = cost_curve(-6, -4, 500)

dev = 91
cost_dev = cost_curve(-5, -2, 101)

purchases = 255
cost_pur = cost_curve(-5, -1, 255)

manufacturing = 37
cost_man = cost_curve(-2, -1, 104)

transport = 10
cost_tran = cost_curve(-1, 0, 10.56)

launches = 7860
cost_laun = cost_curve(-4, 0, 6390)

maintenance = 60
cost_main = cost_curve(10, 15, 65)




#revenues
subsidy1 = 1100
cost_sub1 = cost_curve(-6, 0, 250)

subsidy2 = 250
cost_sub2 = cost_curve(-6, 25, 1100)

revenue = 4930
cost_rev = cost_curve(0, 25, 4930)



#timeline
time = np.arange(-6,25,1)

new_cost = cost_jobs + cost_infra + cost_dev + cost_pur + cost_man + cost_tran + cost_laun + cost_main
new_rev = cost_sub1 +cost_sub2 + cost_rev

timelycost = []
for i in range(0,len(new_cost)):
    total = sum(new_cost[:i])
    timelycost.append(total)

timelyrevs = []
for i in range(0,len(new_rev)):
    total = sum(new_rev[:i])
    timelyrevs.append(total)

plt.figure()
a = np.array(timelycost) 
b = np.array(timelyrevs)
x =  time
plt.plot(x,a,label='Costs')
plt.plot(x,b,label='Income')
plt.ylabel('Millions of Euros')
plt.xlabel('Time')
plt.legend(loc='upper left')
plt.title('Cost and revenue curves')
# idx = np.argwhere(np.diff(np.sign(a - b))).flatten()
# plt.plot(x[idx], a[idx], 'ro')
plt.show()


for i in timelycost:
    print(i/timelycost[-1])


























