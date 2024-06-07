#User Defined KNearestneighbour 
from scipy.spatial import distance
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def euc(a,b):
    return distance.euclidean(a,b)

class Sjpknn():

    def fit(self,Trainingdata,Trainingtarget):
        self.Trainingdata=Trainingdata
        self.Trainingtarget=Trainingtarget

    def predict(self,Testdata):
        predictions=[]
        for row in Testdata:
            label=self.closest(row)
            predictions.append(label)
        return predictions
    
    def closest(self,row):
        bestdistance=euc(row,self.Trainingdata[0])
        bestindex=0
        for i in range (1,len(self.Trainingdata)):
            dist=euc(row,self.Trainingdata[i])
            if (dist<bestdistance):
                bestdistance=dist
                bestindex=i
        
        return self.Trainingtarget[bestindex]
    
def kneighbour():
    border="-"*50
    iris=load_iris()
    data=iris.data
    target=iris.target

    print(border)
    print("Actual data set")
    print(border)

    for i in range(len(iris.target)):
        print("ID :%d, Label%s, Features: %s"%(i,data[i],target[i]))
    print("Size of actual data set %d"%(i+1))

    data_train,data_test,target_train,target_test=train_test_split(data,target,test_size=0.5)

    print(border)
    print("Training data set")
    print(border)
    for i in range(len(data_train)):
        print("ID :%d, Label%s, Features: %s"%(i,data_train[i],target_train[i]))
    print("Size of actual data set %d"%(i+1))


    print(border)
    print("Testing data set")
    print(border)
    for i in range(len(data_test)):
        print("ID :%d, Label%s, Features: %s"%(i,data_test[i],target_test[i]))
    print("Size of actual data set %d"%(i+1))
    print(border)


    classifier = Sjpknn()
    classifier.fit(data_train,target_train)
    predictions=classifier.predict(data_test)
    accuracy=accuracy_score(target_test,predictions)
    return accuracy

def main():
    Accuracy=kneighbour()
    print("Accuracy of classification algorithm with k neighbor classifer is :",Accuracy*100,"%")

if __name__ =="__main__":
    main()

    



    

    