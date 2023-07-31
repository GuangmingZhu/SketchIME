# SketchIME

## Our Annotation Tool
Our developed annotation tool is displayed below. The software displays the standard symbols using a component-incremental style, and displays the free-hand sketches using a stroke-incremental style. The same number is assigned to the sketches if one sketch which contains some strokes matchs one standard symbol above which contains some components. The final semantic component labels will be given based on the above annotation and the symbol's category.

<img src="images/Annotation%20Tool.png" width=80%>

## Some Color-coded Sketch Categories
We visualize some sketch categories using a color-coded style. Different colors in one sketch represent different types of semantic components. But the same color across different types of sketches does not always mean the same semantic component. For example, the big circles in the top two lines and the big rectangles in the bottom two lines are red, but the big circle and the big rectangle are two different semantic components.

<img src="images/Color-coded%20Visualization.png" width=80%>

## Interpretability Analysis
Each sketch category is comprised of some semantic components. An interpretable inference means that our network can recognize one sketch’s category correctly because it also can segment the sketch’s semantic components correctly, vice versa. 

It can be seen from the table below that the segmentation performance of our SketchRecSeg network is extremely high when it recognize the category correctly, but is very low when it does not recognize correctly. This proves that our network can lead to more interpretable inference. 

<img src="images/Original%20Fig%204%20To%20%20Table.png" width=80%>
