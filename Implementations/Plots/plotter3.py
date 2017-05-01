import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
objects=("Logistic regression","KNN","Dec. Tree","ANN","Na.Bayes","SVM")
x=[1,2,3,4,5,6,]
y=[97,90,89,93,88,93]
fig=plt.figure()
ax = fig.add_subplot(111)
plt.plot(x,y,linestyle='--', marker='o', color='b')
plt.axis([0,7,0,120])
plt.xticks(x, objects)
plt.ylabel('F-score')
plt.title('F-scores V/S Method')
for xy in zip(x, y):                                       
    ax.annotate('(%s, %s)' % xy, xy=xy, textcoords='data') 
x1=[1,2,3,4,5,6]
y1=[65,83,65,81,8,65]
ax2 = fig.add_subplot(111)
plt.plot(x1,y1,linestyle='--', marker='o', color='r')
for xy in zip(x1, y1):                                       # <--
    ax2.annotate('(%s, %s)' % xy, xy=xy, textcoords='data') # <--
blue_patch = mpatches.Patch(color='blue', label='F1 score of 1 after under-sampling')  
red_patch = mpatches.Patch(color='red', label='F1 score of 1 before under-sampling')
plt.legend(handles=[red_patch,blue_patch])
plt.show()