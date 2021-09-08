#!/usr/bin/env python
# coding: utf-8

# In[58]:


import matplotlib.pyplot as plt
import numpy as np

from LessonUtil import RandomData


def make_plot():
    fig, axes = plt.subplots()
    axes.scatter([0, 1, 2, 3], [0, 1, 2, 3])


def make_plot():
    fig, axes = plt.subplots()
    axes.scatter([0, 1, 2, 3], [0, 1, 2, 3])
    plt.show()


make_plot()

data = RandomData(60)


def plot_ex1(data):
    fig, axes = plt.subplots()
    axes.scatter(data.x, data.y, c='b')
    plt.show()


plot_ex1(data)


# In[2]:


def plot_ex2(data):
    fig, axes = plt.subplots()
    axes.scatter(data.x, data.y, color='red')
    plt.show()


plot_ex2(data)


# In[3]:


def plot_ex3(data):
    fig, axes = plt.subplots()
    axes.scatter(data.x, data.y, facecolor='g', edgecolor='r')
    # the plural is also available
    # axes.scatter(data.x, data.y, facecolors='g', edgecolors='r')
    plt.show()


plot_ex3(data)


# In[4]:


def plot_ex4(data):
    fig, axes = plt.subplots()
    axes.bar(data.n, data.y, color='b')
    plt.show()


data4 = RandomData(4)
plot_ex4(data4)


# In[5]:


def plot_ex5(data):
    fig, axes = plt.subplots()
    axes.scatter(data.x, data.y, s=150, color=['g', 'r', 'b', 'w'], edgecolor='r')


plt.show()
plot_ex5(data4)


# In[6]:


def plot_ex6(data):
    fig, axes = plt.subplots()
    markers = axes.bar(data.n, data.y, color='red')
    for marker, y in zip(markers, data.y):
        if y > 0:
            marker.set(color='green', linewidth=3)
            marker.set_edgecolor('black')


data10 = RandomData(10)
plot_ex6(data10)


# In[7]:


def plot_ex7(data):
    fig, axes = plt.subplots()
    axes.fill_between(data.n, data.x, data.y, color='red')
    plt.show()


plot_ex7(data10)

# In[11]:


data.n


# In[15]:


def plot_ex8(data):
    fig, axes = plt.subplots()
    axes.fill_between(data.n, data.x, data.y, where=(data.x > data.y), color='c', alpha=0.3, interpolate=True)
    axes.fill_between(data.n, data.x, data.y, where=(data.x <= data.y), color='y', alpha=0.3, interpolate=True)


plot_ex8(data10)

# In[27]:


x, y = np.ogrid[:3, :6]
x

# In[31]:


import itertools


def plot_ex9(data, cc):
    fig, axes = plt.subplots(figsize=(6, 6))
    for i in data.n:
        axes.scatter(i, i, s=200, c=next(cc), edgecolor='purple')
    plt.show()


data = RandomData(40)
colors = itertools.cycle(["r", "b", "g", "yellow"])
plot_ex9(data, colors)

# In[34]:


from cycler import cycler


def plot_ex10(data, cc):
    fig, axes = plt.subplots(figsize=(6, 6))
    axes.set_prop_cycle(cc)
    for i in data.n:
        axes.scatter(i, i)
    plt.show()


cc = cycler(color=['c', 'm', 'y', 'k'])
data = RandomData(40)
plot_ex10(data, cc)


# In[36]:


def plot_ex11(data):
    fig, axes = plt.subplots(figsize=(6, 6))
    for i in data.n:
        axes.scatter(i, i)
    plt.show()


plot_ex11(data)


# In[38]:


def plot_ex12(data):
    fig, axes = plt.subplots()
    axes.fill_between(data.n, data.x, data.y, where=(data.x < data.y), color='C0', alpha=0.3, interpolate=True)
    axes.fill_between(data.n, data.x, data.y, where=(data.x >= data.y), color='C9', alpha=0.5, interpolate=True)
    plt.show()


plot_ex12(data)


# In[39]:


def line_ex():
    t = np.arange(0.0, 5.0, 0.2)
    fig, ax = plt.subplots()
    ax.plot(t, t, '-')
    ax.plot(t, t ** 2, '--')
    ax.plot(t, t ** 3, '>')

    plt.show()
    return fig


fig = line_ex()


# In[41]:


def plot_ex13(data, cc):
    fig, axes = plt.subplots(figsize=(6, 6))
    axes.set_prop_cycle(cc)
    for i in data.n:
        #   axes.plot(i, i) # colors and markers
        axes.scatter(i, i)  # colors only
    plt.show()


cc = cycler(color=['c', 'm', 'y', 'k'], marker=['x', 'o', 'd', 's'])
data = RandomData(40)
plot_ex13(data, cc)


# In[94]:


def plot_ex14(data):
    fig, axes = plt.subplots()
    color = '#0000ff30'
    #  color = (0, 0, 1, 0.20)
    axes.scatter(data.x, data.y, color=color)
    plt.show()


plot_ex14(RandomData(1000))

# In[45]:


from LessonUtil import RandomPetData

data = RandomPetData(100)
print(data.pet)


# In[97]:


def build_colormap():
    return {
        'dog': '#7fc97f',
        'cat': '#beaed4',
        'fish': '#fdc086',
        'n/a': '#ffff99'
    }


build_colormap()

# In[69]:


RandomData(100, 4).c


# In[106]:


def plot_pets(data, color_map):
    fig, ax = plt.subplots()
    for index, x, y, pet_type in zip(data.n, data.x, data.y, data.c):
        ax.scatter(x, y, color=color_map[pet_type], edgecolor='black')
    plt.legend(set(data.c))
    fig.show()
    return fig


number_to_pet = {
    0: 'dog',
    1: 'cat',
    2: 'fish',
    3: 'n/a'
}

pet_data = RandomData(100, 4)
pet_data.c = np.array(list(map(lambda x: number_to_pet[x], pet_data.c)))

pet_data.c
plot_pets(pet_data, build_colormap())

# # overview questions
# 
# It is interesting how can several lines of code turns out to be the scattered point plot. Although the color sections are something I have already known; however, data processing using numpy and matplotlib are something really hard but useful. I want to have more exercises on how to use thos repositories in the next few weeks.
