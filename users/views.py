from django.shortcuts import render
from django.contrib import messages
from sklearn.tree import DecisionTreeClassifier
from users.algorithm.Dropoutpredict import classification
classification = classification()
from sklearn.model_selection import train_test_split
from .forms import UserRegistrationForm


# Create your views here.
def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})


def UserLoginCheck(request):
    return render(request, 'users/UserHome.html', {})
   
   
   
def UserHome(request):
    return render(request, 'users/UserHome.html', {})


def Viewdata(request):
    from django.conf import settings
    import pandas as pd
    import os
    path = settings.MEDIA_ROOT + "\\" + "Bloodlevel_dataset.csv"
    data = pd.read_csv(path)
    # print(data)
    data = data.to_html()
    
    return render(request, "users/Viewdata.html", {"data": data})

def prediction(request):
    if request.method == "POST":
        from django.conf import settings
        import pandas as pd

        LDH = request.POST.get("LDH")
        LDH1 = request.POST.get("LDH1")
        LDH2 = request.POST.get("LDH2")
        LDH3 = request.POST.get("LDH3")
        path = settings.MEDIA_ROOT + "\\" + "Bloodlevel_dataset.csv"
        
        data = pd.read_csv(path)
    
        data = data.drop(["First_Name", "Gender", "Age",'avg'],axis=1)
        
        x = data.iloc[:,0:4]
        print("x:",x)
        
        y = data.iloc[:,-1]
        print("y:",y)
        
        x = pd.get_dummies(x)
        print(x)
        
        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25,random_state = 0)
        x_train = pd.DataFrame(x_train)
        x_test=pd.DataFrame(x_test)
        from sklearn.metrics import confusion_matrix
        ddc = DecisionTreeClassifier()
        print("x_train : ",x_train)
        print('x_test:',x_test)
        test_set = [LDH, LDH1, LDH2, LDH3]
        test_set = pd.Series(test_set).fillna(0)
        print("d:",test_set)
        #print(y_train)
        ddc.fit(x_train, y_train)
        y_pred = ddc.predict([test_set])
        print("y : ", y_pred)
        
        if y_pred==[1]:
            msg = "Ldh may be low level"
            return render(request, 'users/UserAddData.html', {'msg':msg})
        elif y_pred==[0]:
            msg='Ldh will be high level'
            return render (request, 'users/UserAddData.html', {'msg':msg})
        
        else: 
            print("not valid")
            
            return render(request, 'users/UserAddData.html', {})
        
    return render(request, "users/UserAddData.html", {})
        
        
def deeplearning(request):
    loss_and_metrics = classification.DeepLearning()
    loss = loss_and_metrics[0]
    accuracy = loss_and_metrics[1]
    print(loss)
    print(accuracy)
    
    return render(request, "users/classification.html",{"loss" : loss, "accuracy"  :accuracy})


