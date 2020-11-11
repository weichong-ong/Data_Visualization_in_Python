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

## Bivariate Data Exploration
- Quantitative vs Quantitative: **scatterplots** `plt.scatter()`, `sns.regplot()`, **heat maps (2D Histrogram)** `plt.hist2d()`
  - Heat maps are useful in the following cases:
    - To represent a plot for discrete vs. another discrete variable
    - As an alternative to transparency when the data points are enormous
- Quantitative vs Quatitative: **violin plots** `sns.violinplot()`, **box plots*8 `sns.boxplot()`
- Quatitative vs Quatitative: **clustered bar charts**`sns.countplot(data, x, hue)`

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
```
ax = sb.barplot(data = df, x = 'cat_var1', y = 'num_var2', hue = 'cat_var2')
ax.legend(loc = 8, ncol = 3, framealpha = 1, title = 'cat_var2')
```
<p align="center">
  <img src="/images/Adaptations_Bivariate_Plots_Barchart.png" width="400" />
</p>

```
ax = sb.pointplot(data = df, x = 'cat_var1', y = 'num_var2', hue = 'cat_var2',
                  dodge = 0.3, linestyles = "")
```
<p align="center">
  <img src="/images/Adaptations_Bivariate_Plots_Pointplot.png" width="400" />
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
