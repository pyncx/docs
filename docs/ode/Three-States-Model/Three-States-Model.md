
## Ordinary Differential Equation


```python
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
```

![img](img/three-states.png)

### Three State Markov Model

$$\ {\frac{dO(t)}{dt} = k_{io}I(t) + k_{co}.C(t) - (k_{oi} + k_{oc})O(t)}$$ 

$$\ {\frac{dC(t)}{dt} = k_{oc}C(t) + k_{ic}.I(t) - (k_{co} + k_{ci})C(t)}$$ 

$$\ {\frac{dI(t)}{dt} = k_{oi}O(t) + k_{ci}.C(t) - (k_{io} + k_{ic})I(t)}$$ 


```python
k = {"koi":0.5,\
     "kio":0.3,\
     "kco": 0.6,\
     "koc": 0.9,\
     "kic":0.72,\
     "kci":0.8}
```

#### Disregarding Total probability and infinite population of state


```python
# function that returns dy/dt
def model(y,t,k):
    
    '''y[0] = O, y[1] = C, y[2] = I'''
    
    dy1dt = k['kio']*y[2] + k['kco']*y[1] - (k['koi'] + k['koc'])*y[0]
    dy2dt = k['koc']*y[1] + k['kic']*y[2] - (k['kco'] + k['kci'])*y[1]
    dy3dt = k['koi']*y[0] + k['kci']*y[1] - (k['kio'] + k['kic'])*y[2]
    
    dydt = [dy1dt,dy2dt,dy3dt]
    return dydt
```


```python
np.linspace(0,20,20,endpoint=True)
```




    array([ 0.        ,  1.05263158,  2.10526316,  3.15789474,  4.21052632,
            5.26315789,  6.31578947,  7.36842105,  8.42105263,  9.47368421,
           10.52631579, 11.57894737, 12.63157895, 13.68421053, 14.73684211,
           15.78947368, 16.84210526, 17.89473684, 18.94736842, 20.        ])




```python
# initial condition
y0 = [1,0,0]

# time points
t = np.linspace(0,20)


'''solve ODEs'''
y = odeint(model,y0,t,args=(k,))


y1 = np.empty_like(t)
y2 = np.empty_like(t)
y3 = np.empty_like(t)
for i in range(len(t)):
    y1[i] = y[i][0]
    y2[i] = y[i][1]
    y3[i] = y[i][2]
```


```python
# plot results
plt.figure(figsize = [15,5])
plt.plot(t,y1,'r-',linewidth=2,label='open')
plt.plot(t,y2,'b--',linewidth=2,label='closed')
plt.plot(t,y3,'g:',linewidth=2,label='inactive')
plt.xlabel('time')
plt.ylabel('y(t)')
plt.legend()
plt.show()
```


![png](output_10_0.png)


#### Considering total probability = 1

These equations :
    
$$\ {\frac{dO(t)}{dt} = k_{io}I(t) + k_{co}.C(t) - (k_{oi} + k_{oc})O(t)}$$ 

$$\ {\frac{dC(t)}{dt} = k_{oc}C(t) + k_{ic}.I(t) - (k_{co} + k_{ci})C(t)}$$ 

$$\ {\frac{dI(t)}{dt} = k_{oi}O(t) + k_{ci}.C(t) - (k_{io} + k_{ic})I(t)}$$ 

with condition:

$$\ {I = 1 - (O + C)}$$
    
becomes:

$$\ {\frac{dO(t)}{dt} = k_{io} + (k_{co} - k_{io}).C(t) - (k_{oi} + k_{oc} +k_{io})O(t)}$$ 

$$\ {\frac{dC(t)}{dt} = k_{ic} + (k_{oc} - k_{ic}).O(t) - (k_{co} + k_{ci} + k_{ic})C(t)}$$ 



```python
# function that returns dy/dt
def model2(y,t,k):
    
    '''y[0] = O, y[1] = C, y[2] = I'''
    
    dy1dt = k['kio'] +  (k['kco'] - k['kio'])*y[1] - (k['koi'] + k['koc'] + k['kio'])*y[0]
    dy2dt = k['kic'] +  (k['koc'] - k['kic'])*y[0] - (k['kco'] + k['kci'] + k['kic'])*y[1]
    dydt = [dy1dt,dy2dt]
    return dydt
```


```python
N = 1
T =10

# initial condition
y0 = [N,0]

# time points
t = np.linspace(0,T)


'''solve ODEs'''
y = odeint(model2,y0,t,args=(k,))


y1 = np.empty_like(t)
y2 = np.empty_like(t)
for i in range(len(t)):
    y1[i] = y[i][0]
    y2[i] = y[i][1]
    y3[i] = N - y[i][0] - y[i][1]
```


```python
# plot results
plt.figure(figsize = [10,4])
plt.plot(t,y1,'o:',color ="orange",linewidth=2,label='open')
plt.plot(t,y2,'o:',color ="magenta", linewidth=2,label='closed')
plt.plot(t,y3,'o:',color = "blue",linewidth=2,label='inactive')
plt.grid(True)
plt.xlabel('Time(t)',fontsize=15)
plt.ylabel('Fraction of states: y(t)',fontsize=15)
plt.title("Plot of the intermediate solution of the simultaneous diff Eqn",fontsize=15)
plt.legend()
plt.savefig("SDE.png")
plt.show()
```


![png](output_15_0.png)

