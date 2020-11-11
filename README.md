# Data Visualization
## Introduction
I learned more than just how to code to create the visualizations.
I also learned how to design the visualizations from using the right plot type to selecting appropriate variable encoding to create visualizations that are clear not just for me, but also for others.
Summary statistics (Mean and standard deviation) can be misleading: [Anscombe's quartet](https://en.wikipedia.org/wiki/Anscombe%27s_quartet).
It is important to pick out visual patterns, trends and outliers during the analysis.

[Datasaurus](https://www.autodesk.com/research/publications/same-stats-different-graphs)

There are two main reasons for creating visuals using data:

Exploratory analysis is done when you are searching for insights. These visualizations don't need to be perfect. You are using plots to find insights, but they don't need to be aesthetically appealing. You are the consumer of these plots, and you need to be able to find the answer to your questions from these plots.

Explanatory analysis is done when you are providing your results for others. These visualizations need to provide you the emphasis necessary to convey your message. They should be accurate, insightful, and visually appealing.

The five steps of the data analysis process:
1. Extract - Obtain the data from a spreadsheet, SQL, the web, etc.
2. Clean - Here we could use exploratory visuals.
3. Explore - Here we use exploratory visuals.
4. Analyze - Here we might use either exploratory or explanatory visuals.
5. Share - Here is where explanatory visuals live.

## Univariate Data Exploration
- Create bar charts for qualitative variables: `sns.countplot()`, `sns.barplot()`, `plt.bar()`
- Create pie charts: `plt.pie()`
  - A pie chart is a common univariate plot type that is used to depict relative frequencies for levels of a categorical variable. A pie chart is preferably used when the number of categories is less, and you'd like to see the proportion of each category.
- Create histograms for quantitative variables: `plt.hist()`, `sns.distplot()`, `sns.histplot()`

### Scale and Transformation
Certain data distributions will find themselves amenable to scale transformations. The most common example of this is data that follows an approximately log-normal distribution. This is data that, in their natural units, can look highly skewed: lots of points with low values, with a very long tail of data points with large values.
However, after applying a logarithmic transform to the data, the data will follow a normal distribution.

#### Scale the x-axis to log-type, change the axis limits, and increase the x-ticks
```
# Get the ticks for bins between [0 - maximum weight]
bins = 10 ** np.arange(-1, 3+0.1, 0.1)

# Generate the x-ticks you want to apply
ticks = [0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]
# Convert ticks into string values, to be displaye dlong the x-axis
labels = ['{}'.format(v) for v in ticks]

# Plot the histogram
plt.hist(data=pokemon, x='weight', bins=bins);

# The argument in the xscale() represents the axis scale type to apply.
# The possible values are: {"linear", "log", "symlog", "logit", ...}
plt.xscale('log')

# Apply x-ticks
plt.xticks(ticks, labels);
```
<p align="center">
  <img src="/images/Scale_Axis.png" width="400" />
</p>

#### Custom scaling the given data Series, instead of using the built-in log scale
```
def sqrt_trans(x, inverse = False):
    """ transformation helper function """
    if not inverse:
        return np.sqrt(x)
    else:
        return x ** 2

# Bin resizing, to transform the x-axis
bin_edges = np.arange(0, sqrt_trans(pokemon['weight'].max())+1, 1)

# Plot the scaled data
plt.hist(pokemon['weight'].apply(sqrt_trans), bins = bin_edges)

# Identify the tick-locations
tick_locs = np.arange(0, sqrt_trans(pokemon['weight'].max())+10, 10)

# Apply x-ticks
plt.xticks(tick_locs, sqrt_trans(tick_locs, inverse = True).astype(int));
```
<p align="center">
  <img src="/images/Custome_Scale_Series.png" width="400" />
</p>

## Bivariate Data Exploration
Quantitative vs Quantitative: **scatterplots** `plt.scatter()`, `sns.regplot(data, x, y, x_jitter, fit_reg)`, **heat maps (2D Histrogram)** `plt.hist2d(data, x, y, cmin, cmap, bins)`

```
# If truncate=True, the regression line is bounded by the data limits.
# Else if truncate=False, it extends to the x axis limits.
# The x_jitter will make each x value will be adjusted randomly by +/-0.3
# The scatter_kws helps specifying the opaqueness of the data points.
# The alpha take a value between [0-1], where 0 represents transparent, and 1 is opaque.
sb.regplot(data = fuel_econ, x = 'year', y = 'comb', truncate=False, x_jitter=0.3, scatter_kws={'alpha':1/20});
```
<p align="center">
  <img src="/images/Bivariate_Plots_regplot.png" width="400" />
</p>

Heat maps are useful in the following cases:
- To represent a plot for two discrete variables
- As an alternative to transparency when the data points are enormous

```
# Use cmin to set a minimum bound of counts: zero counts will be empty
# Use cmap to reverse the color map.
# Specify bin edges
bins_x = np.arange(0.6, 7+0.3, 0.3)
bins_y = np.arange(12, 58+3, 3)

plt.hist2d(data = fuel_econ, x = 'displ', y = 'comb', cmin=0.5, cmap='viridis_r', bins = [bins_x, bins_y])
plt.colorbar()
plt.xlabel('Displacement (1)')
plt.ylabel('Combined Fuel Eff. (mpg)');
```
<p align="center">
  <img src="/images/Bivariate_Plots_heatmap.png" width="400" />
</p>

```
# Specify bin edges
bins_x = np.arange(0.6, 7+0.7, 0.7)
bins_y = np.arange(12, 58+7, 7)
# Use cmin to set a minimum bound of counts
# Use cmap to reverse the color map.
h2d = plt.hist2d(data = fuel_econ, x = 'displ', y = 'comb', cmin=0.5, cmap='viridis_r', bins = [bins_x, bins_y])

plt.colorbar()
plt.xlabel('Displacement (1)')
plt.ylabel('Combined Fuel Eff. (mpg)');

# Select the bi-dimensional histogram, a 2D array of samples x and y.
# Values in x are histogrammed along the first dimension and
# values in y are histogrammed along the second dimension.
counts = h2d[0]

# Add text annotation on each cell
# Loop through the cell counts and add text annotations for each
for i in range(counts.shape[0]):
    for j in range(counts.shape[1]):
        c = counts[i,j]
        if c >= 100: # increase visibility on darker cells
            plt.text(bins_x[i]+0.5, bins_y[j]+0.5, int(c),
                     ha = 'center', va = 'center', color = 'white')
        elif c > 0:
            plt.text(bins_x[i]+0.5, bins_y[j]+0.5, int(c),
                     ha = 'center', va = 'center', color = 'black')
```
<p align="center">
  <img src="/images/Bivariate_Plots_heatmap_annotation.png" width="400" />
</p>

Quantitative vs Quatitative: **violin plots** `sns.violinplot(data, x, y, inner = 'quartile')`, **box plots** `sns.boxplot(data, x, y)`

```
# Types of sedan cars
sedan_classes = ['Minicompact Cars', 'Subcompact Cars', 'Compact Cars', 'Midsize Cars', 'Large Cars']

# Returns the types for sedan_classes with the categories and orderedness
# Refer - https://pandas.pydata.org/pandas-docs/version/0.23.4/generated/pandas.api.types.CategoricalDtype.html
vclasses = pd.api.types.CategoricalDtype(ordered=True, categories=sedan_classes)

# Use pandas.astype() to convert the "VClass" column from a plain object type into an ordered categorical type
fuel_econ['VClass'] = fuel_econ['VClass'].astype(vclasses);

# TWO PLOTS IN ONE FIGURE
plt.figure(figsize = [16, 5])
base_color = sb.color_palette()[0]

# LEFT plot: violin plot
plt.subplot(1, 2, 1)
#Let's return the axes object
ax1 = sb.violinplot(data=fuel_econ, x='VClass', y='comb', color=base_color, innner='quartile')
plt.xticks(rotation=15);

# RIGHT plot: box plot
plt.subplot(1, 2, 2)
sb.boxplot(data=fuel_econ, x='VClass', y='comb', color=base_color)
plt.xticks(rotation=15);
plt.ylim(ax1.get_ylim()) # set y-axis limits to be same as left plot
```
<p align="center">
  <img src="/images/Bivariate_Plots_violinplot_vs_boxplot.png" width="600" />
</p>

Quatitative vs Quatitative: **clustered bar charts**`sns.countplot(data, x, hue)`

```
sb.countplot(data = fuel_econ, x = 'VClass', hue = 'trans_type')
```
<p align="center">
  <img src="/images/Bivariate_Plots_Clustered_barchart.png" width="400" />
</p>

### Adaptations of Univariate Plots
Histogram of quantitative variable against the qualitative subsets of the data: **faceting**

```
# Convert the "VClass" column from a plain object type into an ordered categorical type
sedan_classes = ['Minicompact Cars', 'Subcompact Cars', 'Compact Cars', 'Midsize Cars', 'Large Cars']
vclasses = pd.api.types.CategoricalDtype(ordered=True, categories=sedan_classes)
fuel_econ['VClass'] = fuel_econ['VClass'].astype(vclasses);

bin_edges = np.arange(12, 58+2, 2)
g = sb.FacetGrid(data = fuel_econ, col = 'VClass', col_wrap=3, sharey=False)
g.map(plt.hist, 'comb', bins = bin_edges);
```
<p align="center">
  <img src="/images/Faceting_Bivariate_num_by_cat.png" width="400" />
</p>

Mean of quantitative variable vs quatitative variable: **adapted bar charts**, **point plots**

```
base_color = sb.color_palette()[0]
sb.barplot(data=fuel_econ, x='comb', y = 'make', order = order, ci = 'sd', color = base_color);
plt.xlabel('Avg. Combined Fuel Efficiency (mpg)')
```
<p align="center">
  <img src="/images/Adapted_Bivariante_barchart.png" width="400" />
</p>

```
sb.pointplot(data=fuel_econ, x='VClass', y='comb', color=base_color, ci='sd', linestyles="")
plt.xticks(rotation=15);
plt.ylabel('Avg. Combined Fuel Efficiency (mpg)')
```
<p align="center">
  <img src="/images/Adapted_Bivariante_pointplot.png" width="400" />
</p>

Mean of quantitative variable vs another quantitative variable: **line plots**

```
# Set a number of bins into which the data will be grouped.
# Set bin edges, and compute center of each bin
bin_edges = np.arange(0.6, 7+0.2, 0.2)
bin_centers = xbin_edges[:-1] + 0.1

# Cut the bin values into discrete intervals. Returns a Series object.
num_var_binned = pd.cut(df['num_var'], bin_edges, include_lowest = True)

# For the points in each bin, compute the mean and standard error of the mean.
mean = df['comb'].groupby(num_var_binned).mean()
std = df['comb'].groupby(num_var_binned).std()

# Plot the summarized data
plt.errorbar(x=bin_centers, y=mean, yerr=std)
```
<p align="center">
  <img src="/images/Adapted_Bivariante_lineplot.png" width="400" />
</p>

## Multivariate Data Exploration
There are four major cases to consider when we want to plot three variables together:

- three numeric variables
- two numeric variables and one categorical variable
- one numeric variable and two categorical variables
- three categorical variables

If we have at least two numeric variables, as in the first two cases, one common method for depicting the data is by using a scatterplot to encode two of the numeric variables, then using a non-positional encoding on the points to convey the value on the third variable, whether numeric or categorical.
Three main non-positional encodings stand out: shape, size, and color. For Matplotlib and Seaborn, color is the easiest of these three encodings to apply for a third variable. Color can be used to encode both qualitative and quantitative data, with different types of color palettes used for different use cases.

### Encoding via Shape

```
cat_markers = [['A', 'o'],
               ['B', 's']]

for cat, marker in cat_markers:
    df_cat = df[df['cat_var1'] == cat]
    plt.scatter(data = df_cat, x = 'num_var1', y = 'num_var2', marker = marker)
plt.legend(['A','B'])
```
<p align="center">
  <img src="/images/Encoding_shape_cat.png" width="400" />
</p>

### Encoding via Size
```
plt.scatter(data = df, x = 'num_var1', y = 'num_var2', s = 'num_var3')

# dummy series for adding legend
sizes = [20, 35, 50]
base_color = sb.color_palette()[0]
legend_obj = []
for s in sizes:
    legend_obj.append(plt.scatter([], [], s = s, color = base_color))
plt.legend(legend_obj, sizes)
```
<p align="center">
  <img src="/images/Encoding_size_num.png" width="400" />
</p>

### Encoding via Color
If we have a **qualitative** variable, we can set different colors for different levels of a categorical variable through the "hue" parameter on seaborn's FacetGrid class.

```
g = sb.FacetGrid(data = df, hue = 'cat_var1', size = 5)
g.map(plt.scatter, 'num_var1', 'num_var2')
g.add_legend()
```
<p align="center">
  <img src="/images/Encoding_color_cat.png" width="400" />
</p>

For **quantitative** variables, we should not take the same approach, since FacetGrid expects any variable input for subsetting to be categorical. Instead, we can set color based on numeric value in the scatter function through the "c" parameter, much like how we set up marker sizes through "s".

```
plt.scatter(data = df, x = 'num_var1', y = 'num_var2', c = 'num_var3')
plt.colorbar()
```
<p align="center">
  <img src="/images/Encoding_color_num.png" width="400" />
</p>

Color Palettes: There are three major classes of color palette to consider:
1. qualitative: nominal data
2. sequential: ordinal and numeric data
3. diverging: ordinal or numeric data with meaningful center point

### Adaptations of Bivariate Plots
#### Faceting for Multivariate Data
- Another way of adding an additional dimension to plot: Faceting bivariate plots by level of third variable -> Multivariate plot

```
# by one qualitative variable
g = sb.FacetGrid(data = df, col = 'cat_var1', size = 4)
g.map(sb.boxplot, 'cat_var2', 'num_var2')
```
<p align="center">
  <img src="/images/Faceting_Multivariate_one_cat.png" width="400" />
</p>

```
by two qualitative variables:
g = sb.FacetGrid(data = df, col = 'cat_var2', row = 'cat_var1', size = 2.5, margin_titles = True)
g.map(plt.scatter, 'num_var1', 'num_var2')
```
<p align="center">
  <img src="/images/Faceting_Multivariate_two_cat.png" width="400" />
</p>

#### Other Adaptations of Bivariate Plots
Mean of a third quantitative variable in a 2-d histogram of two quantitative variables
```
xbin_edges = np.arange(0.25, df['num_var1'].max()+0.5, 0.5)
ybin_edges = np.arange(7,    df['num_var2'].max()+0.5, 0.5)

# count number of points in each bin
xbin_idxs = pd.cut(df['num_var1'], xbin_edges, right = False,
                    include_lowest = True, labels = False).astype(int)
ybin_idxs = pd.cut(df['num_var2'], ybin_edges, right = False,
                    include_lowest = True, labels = False).astype(int)

pts_per_bin = df.groupby([xbin_idxs, ybin_idxs]).size()
pts_per_bin = pts_per_bin.reset_index()
pts_per_bin = pts_per_bin.pivot(index = 'num_var1', columns = 'num_var2').values

z_wts = df['num_var3'] / pts_per_bin[xbin_idxs, ybin_idxs]

# plot the data using the calculated weights
plt.hist2d(data = df, x = 'num_var1', y = 'num_var2', weights = z_wts,
           bins = [xbin_edges, ybin_edges], cmap = 'viridis_r', cmin = 0.5);
plt.xlabel('num_var1')
plt.ylabel('num_var2');
plt.colorbar(label = 'mean(num_var3)');
```
<p align="center">
  <img src="/images/Adaptations_Bivariate_Plots_heatmap_num.png" width="400" />
</p>

Mean of a third quantitative variable in a 2-d histogram of two quatitative variables
```
cat_means = df.groupby(['cat_var1', 'cat_var2']).mean()['num_var2']
cat_means = cat_means.reset_index(name = 'num_var2_avg')
cat_means = cat_means.pivot(index = 'cat_var2', columns = 'cat_var1',
                            values = 'num_var2_avg')
sb.heatmap(cat_means, annot = True, fmt = '.3f',
           cbar_kws = {'label' : 'mean(num_var2)'})
```
<p align="center">
  <img src="/images/Adaptations_Bivariate_Plots_heatmap_cat.png" width="400" />
</p>

An alternative approach for two quatitative variables and one quantitative variable is to adapt a clustered bar chart using the barplot function

Clustered Bar Chart
```
ax = sb.barplot(data = df, x = 'cat_var1', y = 'num_var2', hue = 'cat_var2')
ax.legend(loc = 8, ncol = 3, framealpha = 1, title = 'cat_var2')
```
<p align="center">
  <img src="/images/Adaptations_Bivariate_Plots_Barchart.png" width="400" />
</p>

Clustered Point Plot
```
ax = sb.pointplot(data = df, x = 'cat_var1', y = 'num_var2', hue = 'cat_var2',
                  dodge = 0.3, linestyles = "")
```
<p align="center">
  <img src="/images/Adaptations_Bivariate_Plots_Pointplot.png" width="400" />
</p>

Clustered Box Plot
```
df_sub = fuel_econ.loc[fuel_econ['fuelType'].isin(['Premium Gasoline', 'Regular Gasoline'])]
sb.boxplot(data=df_sub, x = 'VClass', y = 'displ', hue = 'fuelType')
plt.legend(loc = 6, bbox_to_anchor = (1.0, 0.5)) # legend to right of figure
plt.xticks(rotation = 15);
```
<p align="center">
  <img src="/images/Adaptations_Bivariate_Plots_Boxplot.png" width="400" />
</p>

A line plot can be adapted o create frequency polygons for levels of a categorical variable. We create a custom function to send to a FacetGrid object's `map` function that computes the means in each bin, then plots them as lines via `errorbar`.

```
def mean_poly(x, y, bins = 10, **kwargs):
    """ Custom adapted line plot code. """
    # set bin edges if none or int specified
    if type(bins) == int:
        bins = np.linspace(x.min(), x.max(), bins+1)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2

    # compute counts
    data_bins = pd.cut(x, bins, right = False,
                       include_lowest = True)
    means = y.groupby(data_bins).mean()

    # create plot
    plt.errorbar(x = bin_centers, y = means, **kwargs)

bin_edges = np.arange(0.25, df['num_var1'].max()+0.5, 0.5)
g = sb.FacetGrid(data = df, hue = 'cat_var2', size = 5)
g.map(mean_poly, "num_var1", "num_var2", bins = bin_edges)
g.set_ylabels('mean(num_var2)')
g.add_legend()
```
<p align="center">
  <img src="/images/Adaptations_Bivariate_Plots_Lineplot.png" width="400" />
</p>

### Plot Matrices
#### The relationship between the numeric variables in the data
```
g = sb.PairGrid(data = df, vars = ['num_var1', 'num_var2', 'num_var3'])
g.map_diag(plt.hist)
g.map_offdiag(plt.scatter)
```
<p align="center">
  <img src="/images/Plot_Matrices_PairGrid_Hist.png" width="400" />
</p>

#### The relationship between the numeric and categorical variables in the data
```
g = sb.PairGrid(data = df, x_vars = ['num_var1', 'num_var2', 'num_var3'],
                y_vars = ['cat_var1','cat_var2'])
g.map(sb.violinplot, inner = 'quartile')
```
<p align="center">
  <img src="/images/Plot_Matrices_PairGrid_Violin.png" width="400" />
</p>

### Correlation Matrices
For numeric variables, it can be useful to create a correlation matrix as part of your exploration.
```
sb.heatmap(df.corr(), annot = True, fmt = '.2f', cmap = 'vlag_r', center = 0)
```
<p align="center">
  <img src="/images/Plot_Matrices_HeatMap.png" width="400" />
</p>

### Feature Engineering
Feature engineering is all about creating a new variable with a sum, difference, product, or ratio between those original variables that may lend a better insight into the research questions we seek to answer.

Another way that you can perform feature engineering is to use the cut function to divide a numeric variable into ordered bins. When we split a numeric variable into ordinal bins, it opens it up to more visual encodings. For example, we might facet plots by bins of a numeric variable, or use discrete color bins rather than a continuous color scale.
This kind of discretization step might help in storytelling by clearing up noise, allowing the reader to concentrate on major trends in the data. Of course, the bins might also mislead if they're spaced improperly

## Explanatory Visualizations
Explanatory Visualizations are the key component of the communication at the end of an analysis project. A picture is worth a thousand words.
A good explanatory visualization will help us to convey the findings and impress the audience, be it a friend, boss or the world.

Telling stories with data follows these steps:
1. Start with a Question
2. Repetition of the problem is a Good Thing
3. Highlight the Answer
4. Call Your Audience To Action

### Polishing Plots
- Choose an appropriate plot
- Choose appropriate encodings
- Pay attention to design integrity (minimize chart junk and maximize the data-ink ratio)
- Label axes and choose appropriate tick marks
- Provide legends for non-positional variables
- Title the plot and include descriptive comments
  - The main difference between `suptitle` and `title` is that the former sets a title for the Figure as a whole, and the latter for only a single Axes. If using faceting or subplotting, we want to use `suptitle` to set a title for the figure as a whole.

```
# loading in the data, sampling to reduce points plotted
fuel_econ = pd.read_csv('./data/fuel_econ.csv')

np.random.seed(2018)
sample = np.random.choice(fuel_econ.shape[0], 200, replace = False)
fuel_econ_subset = fuel_econ.loc[sample]

# plotting the data
plt.figure(figsize = [7,4])
plt.scatter(data = fuel_econ_subset, x = 'displ', y = 'comb', c = 'co2',
            cmap = 'viridis_r')
plt.title('Fuel Efficiency and CO2 Output by Engine Size')
plt.xlabel('Displacement (l)')
plt.ylabel('Combined Fuel Eff. (mpg)')
plt.colorbar(label = 'CO2 (g/mi)');
```
<p align="center">
  <img src="/images/Polished_Plot.png" width="400" />
</p>

```
# set up a dictionary to map types to colors
type_colors = {'fairy': '#ee99ac', 'dragon': '#7038f8'}

# plotting
g = sb.FacetGrid(data = pokemon_sub, hue = 'type', size = 5,
                 palette = type_colors)
g.map(plt.scatter, 'weight','height')
g.set(xscale = 'log') # need to set scaling before customizing ticks
x_ticks = [0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]
g.set(xticks = x_ticks, xticklabels = x_ticks)

# add labels and titles
g.set_xlabels('Weight (kg)')
g.set_ylabels('Height (m)')
plt.title('Heights and Weights for Fairy- and Dragon-type Pokemon')
plt.legend(['Fairy', 'Dragon'], title = 'Pokemon Type');
```
<p align="center">
  <img src="/images/Polished_Plot_2.png" width="400" />
</p>

<!--
Course Structure
As this course covers a broad range of ways that data visualizations can be used in the data analysis process, there will also be a large number of topics that will be touched upon. Below is a summary of topics that will be covered in the remaining lessons in this course.

Lesson 2: Design of Visualizations
Before getting into the actual creation of visualizations later in the course, this lesson introduces design principles that will be useful both in exploratory and explanatory analysis. You will learn about different data types and ways of encoding data. You will also learn about properties of visualizations that can impact both the clarity of messaging as well as their accuracy.

Lessons 3-5: Exploration of Data
These lessons systematically present core visualizations in exploratory data analysis. Exploration starts with univariate visualizations to identify trends in distribution and outliers in single variables. Bivariate visualizations follow, to show relationships between variables in the data. Finally, multivariate visualization techniques are presented to identify complex relationships between three or more variables at the same time.

Lesson 6: Explanatory Visualizations
This lesson describes considerations that should be made when moving from exploratory data analysis to explanatory analysis. When polishing visualizations to present to others, you will need to consider what findings you want to focus on and how to use visualization techniques to highlight your main story. This lesson also provides tips for presentation of results and how to iterate on your presentations.

Lesson 7: Visualization Case Study
In this lesson, you will bring together everything from the previous lessons in an example case study. You will be presented with a dataset and perform an exploratory analysis. You will then take findings from that analysis and polish them up for presentation as explanatory visualizations.
-->
