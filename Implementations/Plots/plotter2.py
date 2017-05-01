import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
objects=("Logistic regression","KNN","Dec. Tree","ANN","Na.Bayes","Proposed Technique")
x=[1,2,3,4,5,6]
y=[99.92,99.96,99.09,99.95,97.7,98.80]
fig=plt.figure()
ax = fig.add_subplot(111)
plt.plot(x,y,linestyle='--', marker='o', color='b')
plt.axis([0,7,0,100])
plt.xticks(x, objects)
plt.ylabel('Accuracy')
plt.title('Accuracy V/S Method')
for xy in zip(x, y):                                       
    ax.annotate('(%s, %s)' % xy, xy=xy, textcoords='data') 
x1=[1,2,3,4,5,6]
y1=[95.17,94.01,92.1,95.21,94.05,96.17]
ax2 = fig.add_subplot(111)
plt.plot(x1,y1,linestyle='--', marker='o', color='r')
plt.axis([0,7,0,120])
plt.xticks(x1, objects)
plt.ylabel('Accuracy')
plt.title('Accuracy V/S Method')
for xy in zip(x1, y1):                                       # <--
    ax2.annotate('(%s, %s)' % xy, xy=xy, textcoords='data') # <--
blue_patch = mpatches.Patch(color='blue', label='Initial data set')  
red_patch = mpatches.Patch(color='red', label='The undersamples after SMOTE data')
plt.legend(handles=[red_patch,blue_patch])
plt.show()