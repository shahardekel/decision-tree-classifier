import graphviz
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
#from graphviz import render
from sklearn.metrics import accuracy_score
import numpy as np
from anytree import NodeMixin
from anytree.exporter import DotExporter
from sklearn.metrics import confusion_matrix
import seaborn as sn


#A
X, y = datasets.load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 3)
cancer=load_breast_cancer()


#B
model=tree.DecisionTreeClassifier(criterion="entropy",random_state=2)
model= model.fit(X_train, y_train)

#for running in pycharm
tree.plot_tree(model)
plt.title('sklearn decision tree')
plt.show()

#for running in google colab
"""dot_data = tree.export_graphviz(model, out_file=None,feature_names=cancer.feature_names,
                     class_names=cancer.target_names,  filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph.render('dot', 'png', 'my decision tree.dot')
graph"""

y_train_pred_skl=model.predict(X_train)
y_test_pred_skl=model.predict(X_test)
train_accuracy=accuracy_score(y_train, y_train_pred_skl)
print("sklearn's train accuracy:",train_accuracy)
test_accuracy=accuracy_score(y_test, y_test_pred_skl)
print("sklearn's test accuracy:",test_accuracy)

#C
#we want the data and target on the same vector so we won't have to deal with both X_train and y_train
def make_vector(X,y):
    result=np.c_[X,y]
    return result


#to make a tree->we need the entropy measurement->we will build a function that gets T(group) and calculate the entropy
def find_entropy(one_vector):
    malignant=0
    benign=0
    for i in one_vector:
        if i[-1] == 0:
            malignant+=1
        elif i[-1] == 1:
            benign += 1
    group_size = malignant+benign
    #print(group_size)
    # initial condition->we don't want negative probability or divide by 0
    if malignant<=0 or benign<=0:
        return 0
    else:
        entropy= -(((malignant/group_size)*np.log2(malignant/group_size))+((benign/group_size)*np.log2(benign/group_size)))
    return entropy


#we want to divide the T group to Tleft and Tright by a treshold (according the tutorial)
def splitT(one_vec, treshold, index):
    Tright, Tleft=list(),list()
    for r in one_vec:
        if r[index]<=treshold:
            Tleft.append(r)
        else:
            #if r[index]>treshold:
            Tright.append(r)
    return Tleft,Tright


#now we want to take the best split according to the function J(T,k,threshold) we saw in the tutorial
def J_func(Tleft,Tright):
    #Tleft,Tright=splitT(one_vec,treshold, index)
    mleft=len(Tleft)
    mright=len(Tright)
    m=mleft+mright
    imp_left=find_entropy(Tleft)
    imp_right=find_entropy(Tright)

    J_res=((mleft/m)*imp_left)+((mright/m)*imp_right)
    return J_res


#we want to find the best split->the function will get the vector and returns the best condition to split according to the smallest J function
def best_split(one_vec):
    #we need to check if we reached a leaf
    M_class=False
    B_class=False
    for item in one_vec:
        if item[-1]==0:
            M_class=True
        else:
            B_class=True
    if M_class==False or B_class==False: # if we reach a leaf
        return None,None
    result=[None,None]
    best_J=1 #J is a probability function->it cannot be more than 1
    #checking for split according to J(T,k,threshold) we saw in the tutorial
    for index in range(len(one_vec[0]) - 1):
        for row in one_vec:
            Tleft, Tright=splitT(one_vec,row[index],index)
            J_tmp=J_func(Tleft,Tright)
            if J_tmp<best_J:
                best_J=J_tmp
                result=[index,row[index]]

    return result


#now we want to build a node that holds the condition for the split, the entropy, number of samples,
#and the value for each sample(malignant(0) or benign(1))
class Node(NodeMixin): #we use the anytree library
    def __init__(self, entropy, samples, value, children=None, parent=None):
        self.condition=None
        self.entropy=entropy
        self.samples=samples
        self.value = value
        self.threshold=0
        self.label=0
        if value[0]>value[1]:
            chosen_class=0 #Malignant
        else:
            chosen_class=1 #Benign
        self.pred=chosen_class #for accuracy check
        if children:
            self.children=[children]
        if parent:
            self.parent=parent

#for the value vector
def count_values(one_vec):
    malignant,benign = 0,0
    for item in one_vec:
        if item[-1] == 0: #checks the last place in one_vec(for the label)
            malignant += 1
        elif item[-1] == 1:
            benign += 1
    return [malignant,benign]


def make_tree(node,one_vec,label):
    condition_index, condition_value=best_split(one_vec)
    if node.entropy!=0: #exit condition
        Tleft,Tright=splitT(one_vec,condition_value,condition_index)
        node.label=condition_index
        node.threshold=condition_value
        node.condition=f'{label[condition_index]}<={condition_value}'

        left_child_values=count_values(Tleft)
        right_child_values=count_values(Tright)
        left_child_entropy=find_entropy(Tleft)
        right_child_entropy=find_entropy(Tright)

        #children making
        left_node=Node(entropy=left_child_entropy,samples=left_child_values[0]+left_child_values[1], value=left_child_values,parent=node)
        right_node = Node(entropy=right_child_entropy, samples=right_child_values[0] + right_child_values[1],value=right_child_values, parent=node)
        node.children=[left_node,right_node]

        #going to build the tree in both left and right children
        if left_child_entropy !=0:
            make_tree(left_node,Tleft,label)
        if right_child_entropy !=0:
            make_tree(right_node,Tright,label)
    return


label=cancer.feature_names
one_vec_train=make_vector(X_train,y_train)
one_vec_test=make_vector(X_test,y_test)
init_entropy=find_entropy(one_vec_train)
root_values=count_values(one_vec_train)
root=Node(entropy=init_entropy,samples=root_values[0]+root_values[1],value=root_values)
make_tree(root,one_vec_train,label)

#after we built the tree, we want to find its accuracy on the train data and the test data
def choose_class(init_node,data_vec):
    class_prediction=[]
    for item in data_vec:
        next_node = init_node
        while not next_node.is_leaf: #goes down the tree
            if item[next_node.label]<=next_node.threshold: #Satisfies the condition for the left child (as seen in class)
                next_node=next_node.children[0]
            else:
                next_node=next_node.children[1] #goes to the right child

        class_prediction.append(next_node.pred)
    #print(class_prediction)
    return class_prediction

#D
#find the train and test accuracies
train_pred=choose_class(root,one_vec_train)
test_pred=choose_class(root,one_vec_test)
train_accuracy=accuracy_score(y_train, train_pred)
print("my algorithm train accuracy:",train_accuracy)
test_accuracy=accuracy_score(y_test, test_pred)
print("my algorithm test accuracy:",test_accuracy)

#plot the tree as a text file
#we plot the tree on the website: https://dreampuf.github.io/GraphvizOnline/ (copy the text in the text file and paste it there)
def plot_tree(root,classes):
    def nodenamefunc(node):
        if classes[node.pred]==0:
            classes[node.pred]="Malignant"
        else:
            classes[node.pred] ="Benign"
        return '%s\nentropy=%s\nsamples=%s\nvalues=%s\nclass=%s' % \
               (node.condition, node.entropy, node.samples, node.value, classes[node.pred])

    def edgeattrfunc(node, child):
        return f'label={True if child == node.children[0] else False}'

    def edgetypefunc(node, child):
        return '--'

    # write the graph visualization to file
    with open('graph.txt', 'w') as file:
        data=DotExporter(root, graph="graph", nodenamefunc=nodenamefunc, nodeattrfunc=lambda node: "shape=box",
                                edgeattrfunc=edgeattrfunc, edgetypefunc=edgetypefunc)
        for line in data:
            file.write(line)

plot_tree(root, [0, 1])

#E
#plot the unnormalized confusion matrix
#for my algorithm
cm_train=confusion_matrix(y_train,train_pred)
cm_test=confusion_matrix(y_test,test_pred)

sn.heatmap(cm_train, annot=True, cmap="tab20",fmt='g')
plt.title('my algorithm- train data')
plt.show()

sn.heatmap(cm_test, annot=True, cmap="tab20",fmt='g')
plt.title('my algorithm- test data')
plt.show()

#for sklearn's algorithm
skl_cm_train=confusion_matrix(y_train, y_train_pred_skl)
skl_cm_test=confusion_matrix(y_test, y_test_pred_skl)

sn.heatmap(skl_cm_train, annot=True, cmap="tab20",fmt='g')
plt.title('sklearn algorithm- train data')
plt.show()

sn.heatmap(skl_cm_test, annot=True, cmap="tab20",fmt='g')
plt.title('sklearn algorithm- test data')
plt.show()


########## RESULTS ##########
"""
sklearn's train accuracy: 1.0
sklearn's test accuracy: 0.8947368421052632
my algorithm train accuracy: 1.0
my algorithm test accuracy: 0.9035087719298246
"""























