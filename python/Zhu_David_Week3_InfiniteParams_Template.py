#!/usr/bin/env python
# coding: utf-8

# In[3]:


def name_value_pairs(**kargs):
    # access kargs like a dictionary
    for k,v in kargs.items():
        print(k,v)
    # just the keys please
    for k in kargs:
        print(kargs[k])
    # ask for a specific value
    print(kargs['classname'])

name_value_pairs(classname = 'Info 490',
                 credits = 3,
                 on_line = True)


# In[7]:


def add_me(*args): 
    v= 0
    for e in args:
        v += e  # same as v = v + e
    return v


# In[8]:


def simple_unpack_demo():
    numbers = [x for x in range(1,10)]
    print(numbers)
    print(add_me(*numbers))

simple_unpack_demo()


# In[10]:


def who_am_i(**kargs):
    for k,v in kargs.items():
        print(k,v)
        
me = { 'classname': 'Info 490',
       'credits':    3,
       'on_line':    True}

print("1",  me)
print("2", *me) # pass in keys only
who_am_i(**me)


# In[12]:


def demo_unpack_list():
    my_items = [1,2,3,"apple"]
    print(my_items)
    print(*my_items)

demo_unpack_list()


# In[23]:


values = (10,11,12,13)
print('{3:04d}'.format(*values))


# In[24]:


# 1. What are positional arguments? they are various lengthed arguments that can be passes as a parameter
# 2. How * and ** operate on lists and dictionaries? * stands for positional arguments and ** stands for positional key/value pairs
# 3. What must you do if you pass in additional argument where the first argument is *? the rest should be named arguments (or key/value pairs)


# In[46]:


def multiply_me(*args):
#     print(args == ())
    if args == ():
        return 0
    result = 1
    for arg in args:
        result *= arg
    return result

multiply_me(1, 2, 3, 4)


# In[63]:


def equation(numbers):
    eq = ""
    if type(numbers) != list:
        return None
    for number in numbers:
#         if type(number) != 
        eq += ' {} *'.format(str(number))
    eq = eq[1: -1] + '= {}'.format(str(multiply_me(*numbers)))
    return eq


# In[64]:


equation([1, 2, 4, 6, 3, 5, 6, 6])


# In[ ]:




