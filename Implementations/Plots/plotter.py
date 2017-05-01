import matplotlib.pyplot as plt
objects=("Logistic regression","KNN","Dec. Tree","ANN","Na.Bayes","SVM","Proposed Technique")
x=[1,2,3,4,5,6,7]
y=[82.25,82.76,82.25,81.67,75.07,83.57,83.11]
fig=plt.figure()
ax = fig.add_subplot(111)
plt.plot(x,y,linestyle='-', marker='o', color='b')
plt.axis([0,8,0,100])
plt.xticks(x, objects)
plt.ylabel('Accuracy')
plt.title('Accuracy V/S Method')
for xy in zip(x, y):                                       # <--
    ax.annotate('(%s, %s)' % xy, xy=xy, textcoords='data') # <--
plt.show()