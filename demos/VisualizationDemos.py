#!/usr/bin/env python
# coding: utf-8

# # Visualization Lab

# For our applications, the built-in Pandas plotting functionality, Seaborn package, and Matplotlib package will be the main ways we plot data.  While there is a lot of overlap between the capabilities each package, they serve different roles.  Here, we will look at how these three package accomplish the same task.  In the lab we employ Matplotlib to greater depth, but we first want to see how easy it is make common plots with Pandas and Seaborn.  First, we will make simple plots of the S&P 500 in 2018.  We will then generate some data and make multiline plots, histograms, and box-and-whisker plots with each of the three libraries.  For reference, a simple image is created with Matplotlib and the interactive functionality of Holoviews is included.

# In[4]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


# # Simple Line Plots

# With a new dataset we commonly need to know the type of data we have and how it looks.  With the S&P 500 dataset we first need to know what columns we have and the type of data it contains.  With `head` we can produce a visualization that is effectively a table that contains the column names and the first rows, which the default gives 5.  If we are interested in seeing the behavior of the "`Adj Close`" column it is trivial to produce a simple plot.  Even though the line graph is basic and lacks things like a y-label, for our rapid exploration it serves its purpose.  It is also worth noting that Pandas plots, as well as Seaborn and Matplotlib, handle dates well.

# In[10]:


data = pd.read_csv('../data/stock_data.csv', index_col=0, parse_dates=True)

print(data.head())# output chart of columns and first few lines
data['Adj Close'].plot();               # quickly plot 'Adj Close' columns


# We can make a similar plot with Seaborn.  Seaborn was designed to handle dataframes well so we can simply tell it what dataframe to use as its input data and then what columns to plot.  We can either pass it directly (`data.index`) or by name ("`Adj Close`").  While Seaborn still produces the plot in one line, it is slightly longer.  However, the plot it produces looks a little more refined, for instance it now automatically generates a y-label.  This is a plot you could pass along to a colleague, but it still isn't perfect since the limits of the x-axis are slightly off and the plot would look better with a thicker line and a descriptive title.

# In[18]:


sns.lineplot(x = data.index, y = 'Low', data = data);


# With Matplotlib we have maximum control over our plots.  It is important to note that since Pandas plots and Seaborn plots are producing Matplotlib plots these tweaks can be applied to them, but if you are going to change a lot just making them is plain Matplotlib may be easier.  Now we can add a descriptive title, change the tick frequency and labels on the x-axis to be more meaningful and easily thicken the line.  It takes a few lines of code, but in the end we can produce a plot that is ready to be shared with a key stakeholder with little effort.

# In[25]:


plt.plot(data.index, data['Low'], linewidth=1)
plt.plot(data.index, data['High'], linewidth=1)
plt.plot(data.index, data['Open'], linewidth=1)
plt.xlim([data.index[0], data.index[-1]])
plt.xticks(['2018-01', '2018-04', '2018-08', '2018-12'], ['January', 'April', 'August', 'December'])
plt.xlabel('Date')
plt.ylabel('Adjusted Close')
plt.title('S&P 500 in 2018');


# # Library Comparison

# Making visualization in data analysis extends far beyond a simple line plot.  Let's now compare how Pandas, Seaborn, and Matplotlib make multiline plots, histograms, and box-and-whisker plots "off the shelf" with no input from additional libraries.  

# Let's generate 4 different series of 1000 data points each and store them in a dataframe.

# In[88]:


df = pd.DataFrame(np.random.randn(1000, 5), columns=['A1','A2','A3','A4','A5']).cumsum()


# ### Pandas

# With the `plot` method we produce a nice looking graph of the 4 data series in 11 key strokes.  All the lines are different colors which aids in interpretation and Pandas automatically produces a label to further help remove ambiguity.  However, labels on the x and y axes are missing.

# In[89]:


df.plot();


# Histograms of the dataframe are produced with similar ease and are automatically produced in four subplots with titles denoting the data column.  Again, the axes aren't labeled, but it couldn't be simpler to make these plots.

# In[34]:


df.hist();


# Box-and-whisker plots show the distributions of the four data columns by marking the 4 quartiles.  The plot uses the Matplotlib defaults and labels each box with the column name on the x-axis.

# In[35]:


df.plot.box();


# ### Seaborn

# In Seaborn, `lineplot` can only plot one series at a time.  Therefore, to get the four lines plotted on the same plot like we want, we just need to call it four times and pass the different columns we want to plot.  The plot looks similar to the Pandas plot, but the y-axis label is off since it is trying to be clever and do it automatically for us.  If we were to only plot one line, this would be a useful addition to the plot.

# In[38]:


sns.lineplot(df.index,df['A'])
sns.lineplot(df.index,df['B'])
sns.lineplot(df.index,df['C'])
sns.lineplot(df.index,df['D'])
sns.lineplot(df.index,df['E']);


# Seaborn excels at plotting a histogram and offers additional functionality that would normally require additional work by the user.  Making the histogram is easy with a simple passing of the data series to `distplot` and it produces a histogram with a kernel density estimation overlaid.  This is useful because it can tell us a bit more about the data within the histogram bins.

# In[10]:


sns.distplot(df['A']);


# To make the box-and-whisker plot, we first need to use `melt` to turn the class identifiers of our columns into a categorical variable.  One perk of Seaborn is that is clever enough to discern categorical variables and group data points.  Usually, this is a benefit, but the way our data is setup requires us to make this one quick reformatting before passing it to the `boxplot` function where the categorical classes are identified.  The plot it produces looks nicer than the Pandas plot and the different colors help to distinguish the different classes.  Additionally, the axes labels are automatically generated. 

# In[11]:


df_melt = pd.melt(df,var_name='class', value_name='value')


# In[12]:


sns.boxplot(x='class', y='value', data=df_melt);


# ### Matplotlib

# Matplotlib offers the control to hone and refine visualizations so they deliver their maximum impact.  When plotting the lines of our data, we explicitly plot each line, set the axes labels, the plot limits, the tick frequency, linewidths and place the legend where we want it.  If we wanted, we could also change the plot styles and many other attributes.

# In[13]:


fig = plt.figure()
fig, ax = plt.subplots(1, 1)

ax.plot(df.index, df['A'], linewidth=2, label='Line 1')
ax.plot(df.index, df['B'], linewidth=2, label='Line 2')
ax.plot(df.index, df['C'], linewidth=2, label='Line 3')
ax.plot(df.index, df['D'], linewidth=2, label='Line 4')

ax.set_ylabel('y values')
ax.set_xlabel('x values')
ax.set_xlim([df.index[0], df.index[-1]])
ax.set_ylim([df.values.min(), df.values.max()])
ax.set_xticks([0, 250, 500, 750, 1000])

ax.legend(loc='upper right')

plt.title('Matplotlib Plot');


# Using subplots to contain graphs allows you to separate information for easier interpretation.  With Matplotlib we can do this easily, but it is not built into Pandas or Seaborn.  Although, since Pandas plotting and Seaborn plotting are just wrappers for Matplotlib and produce Matplotlib axes, we can create axes in subplots and pass them to functions from these libraries.

# In[43]:


fig = plt.figure()
fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
ax[0,0].plot(df.index.values, df['A'], label='Line 1')
ax[0,1].plot(df.index.values, df['B'], label='Line 2')
ax[1,0].plot(df.index.values, df['C'], label='Line 3')
ax[1,1].plot(df.index.values, df['D'], label='Line 4')
plt.subplots_adjust(wspace=0.0, hspace=0.0)
fig.text(0.5, 0.00, 'common X', ha='center')
fig.text(0.00, 0.5, 'common Y', va='center', rotation='vertical');


# In the histogram below we can see how much is available to customize.  Not only can we improve the axes and titles, we can also add informative text to share relevant information.  For demonstration purposes we can color code the bars according to their height, although it muddles the plot by mixing two visual cues to encode the same information (i.e. using both length and color for the value of the frequency).

# In[50]:


from matplotlib import colors

fig, ax = plt.subplots(1,1)
n, bins, patches = plt.hist(df['B'], bins='auto', edgecolor='pink')
fracs = n/n.max()
norm = colors.Normalize(fracs.min(), fracs.max())

for f, p in zip(fracs, patches):
    color = plt.cm.viridis(norm(f))
    p.set_facecolor(color)
    
mean = np.mean(df['B'])
    
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Modified Matplotlib Slinus Histogram')
plt.text(0.7, 0.9, f'$\mu={mean:.2f}$', transform=ax.transAxes);


# A box-and-whisker plot takes a fair amount of more work than the other two methods that only required one line, but we can do things like change the extension to colors other than the main boxes if we wanted, or color code each individual box to a meaningful color, rather than the default colors produced by Seaborn.

# In[16]:


def draw_plot(data, edge_color, fill_color):
    bp = ax.boxplot(data, patch_artist=True)

    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color=edge_color)

    for patch in bp['boxes']:
        patch.set(facecolor=fill_color) 


# In[17]:


fig, ax = plt.subplots()
draw_plot(df.values, 'black', 'steelblue')
ax.set_xticklabels(['Distribution1', 'Distribution2', 'Distribution3', 'Distribution4'],
                    rotation=45, fontsize=8);
ax.set_ylabel('y values')
plt.title('Matplotlib Box Plot');


# ### Image Plotting

# Matplotlib also makes plotting images convenient.

# In[18]:


import matplotlib.image as mpimg
logo = mpimg.imread('../data/mk_circle.jpg')
plt.imshow(logo)


# ### Holoviews

# Interactive visualizations are a great way to accelerate your own data exploration or augment your message when sharing your work.  Often subtle details and relationships are hard to tease from static plots.  In the early stages of understanding a new data source interactivity allows you to quickly pose questions and explore hypotheses.  Additionally, tools like "sliders" for showing time variable data allow you to see trends easily and "lassos" allow you to quickly isolate subgroups of data from a larger population to understand their characteristics.  When sharing your work, a well-designed interactive console guides people through your "data story" by allowing them to explore the data themselves.

# With the added punch of an interactive visualization comes additional work.  The Holoviews gallery offers many examples of what can be done with the package.

# In[69]:


"""
Example app demonstrating how to use the HoloViews API to generate
a bokeh app with complex interactivity. Uses a RangeXY stream to allow
interactive exploration of the mandelbrot set.
"""

import numpy as np
import holoviews as hv

from holoviews import opts
from holoviews.streams import RangeXY
from numba import jit
from bokeh.io import output_notebook
#from bokeh.plotting import show
#import bokeh.plotting as bkh
output_notebook()

hv.extension('bokeh')

renderer = hv.renderer('bokeh')
renderer = renderer.instance(mode='server')

@jit
def mandel(x, y, max_iters):
    """
    Given the real and imaginary parts of a complex number,
    determine if it is a candidate for membership in the Mandelbrot
    set given a fixed number of iterations.
    """
    i = 0
    c = complex(x,y)
    z = 0.0j
    for i in range(max_iters):
        z = z*z + c
        if (z.real*z.real + z.imag*z.imag) >= 4:
            return i

    return 255

@jit
def create_fractal(min_x, max_x, min_y, max_y, image, iters):
    height = image.shape[0]
    width = image.shape[1]

    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_y - min_y) / height
    for x in range(width):
        real = min_x + x * pixel_size_x
        for y in range(height):
            imag = min_y + y * pixel_size_y
            color = mandel(real, imag, iters)
            image[y, x] = color

    return image

def get_fractal(x_range, y_range):
    (x0, x1), (y0, y1) = x_range, y_range
    image = np.zeros((600, 600), dtype=np.uint8)
    return hv.Image(create_fractal(x0, x1, -y1, -y0, image, 200),
                    bounds=(x0, y0, x1, y1))

# Define stream linked to axis XY-range
range_stream = RangeXY(x_range=(-1., 1.), y_range=(-1., 1.))

# Create DynamicMap to compute fractal per zoom range and
# adjoin a logarithmic histogram
dmap = hv.DynamicMap(get_fractal, label='Manderbrot Explorer',
                     streams=[range_stream]).hist(log=True)

# Apply options
dmap.opts(
    opts.Histogram(framewise=True, logy=True, width=100),
    opts.Image(cmap='Greens', logz=True, height=200, width=200,
               xaxis=1, yaxis=1))

doc = renderer.server_doc(dmap)
doc.title = 'Mandelbrot Explorer'
dmap


# In[77]:


#app = renderer.app(dmap)
#from bokeh.server.server import Server
#server = Server({'/': app}, port=0)
#server.start()
#server.show('/');

import numpy as np
import holoviews as hv
hv.extension('bokeh')
frequencies = [0.5, 0.75, 1.0, 1.25]

def sine_curve(phase, freq):
    xvals = [0.1* i for i in range(100)]
    return hv.Curve((xvals, [np.sin(phase+freq*x) for x in xvals]))

# When run live, this cell's output should match the behavior of the GIF below
dmap = hv.DynamicMap(sine_curve, kdims=['phase', 'frequency'])
dmap.redim.range(phase=(0.8,1), frequency=(0.8,1.25))


# In[ ]:





# In[ ]:




