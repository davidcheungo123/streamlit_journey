import streamlit as st 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC


@st.cache
def get_data_SVM(data_num):
    url = f"./data/data{str(data_num)}.txt"
    return np.loadtxt(url)

def learn_and_display_SVM(datafile, kernel_type='rbf', C_value=1.0, s_value=1.0):
    data = datafile
    n,d = data.shape
    # Create training set x and labels y
    x = data[:,0:2]
    y = data[:,2]
    # Now train a support vector machine and identify the support vectors
    if kernel_type == 'rbf':
        clf = SVC(kernel='rbf', C=C_value, gamma=1.0/(s_value*s_value))
    if kernel_type == 'quadratic':
        clf = SVC(kernel='poly', degree=2, C=C_value, coef0=1.0)
    clf.fit(x,y)
    sv = np.zeros(n,dtype=bool)
    sv[clf.support_] = True
    notsv = np.logical_not(sv)
    # Determine the x1- and x2- limits of the plot
    x1min = min(x[:,0]) - 1
    x1max = max(x[:,0]) + 1
    x2min = min(x[:,1]) - 1
    x2max = max(x[:,1]) + 1
    fig, ax = plt.subplots()
    plt.xlim(x1min,x1max)
    plt.ylim(x2min,x2max)
    # Plot the data points, enlarging those that are support vectors
    plt.plot(x[(y==1)*notsv,0], x[(y==1)*notsv,1], 'ro')
    plt.plot(x[(y==1)*sv,0], x[(y==1)*sv,1], 'ro', markersize=10)
    plt.plot(x[(y==-1)*notsv,0], x[(y==-1)*notsv,1], 'k^')
    plt.plot(x[(y==-1)*sv,0], x[(y==-1)*sv,1], 'k^', markersize=10)
    # Construct a grid of points and evaluate classifier at each grid points
    grid_spacing = 0.05
    xx1, xx2 = np.meshgrid(np.arange(x1min, x1max, grid_spacing), np.arange(x2min, x2max, grid_spacing))
    grid = np.c_[xx1.ravel(), xx2.ravel()]
    Z = clf.decision_function(grid)
    # Quantize the values to -1, -0.5, 0, 0.5, 1 for display purposes
    for i in range(len(Z)):
        Z[i] = min(Z[i],1.0)
        Z[i] = max(Z[i],-1.0)
        if (Z[i] > 0.0) and (Z[i] < 1.0):
            Z[i] = 0.5
        if (Z[i] < 0.0) and (Z[i] > -1.0):
            Z[i] = -0.5
    # Show boundary and margin using a color plot
    Z = Z.reshape(xx1.shape)
    plt.pcolormesh(xx1, xx2, Z, cmap=plt.cm.PRGn, vmin=-2, vmax=2)
    st.pyplot(fig)

def rand2sym(vec, cdf, sym):
    """ Transform an array of random numbers, distributed uniformly in [0,1]
    into a sequence of symbols, chosen according to the probabilities defined by c (cumul of p)"""
    ans=[]
    counts={i:0 for i in range(4)}
    for x in vec:
        for i in range(len(cdf)-1):
            if x>=cdf[i] and x<cdf[i+1]:
                ans.append(sym[i])
                counts[i]+=1
                break
    return ans,counts

def probability_stimulation(n):
    # red_bck="\x1b[41m%s\x1b[0m"
    # green_bck="\x1b[42m%s\x1b[0m"
    # tan_bck="\x1b[43m%s\x1b[0m"
    # blue_bck="\x1b[44m%s\x1b[0m"
    # sym=[red_bck%'6',green_bck%'1',tan_bck%'3',blue_bck%'4']
    sym=["<font color=‘violet’>1</font>\n","<font color=‘green’>2</font>\n","<font color=‘orange’>3</font>\n","<font color=‘blue’>4</font>\n"]
    p=[0.0,0.1,0.2,0.3,0.4]
    c= list(np.cumsum(p))
    R=np.random.rand(n)
    _syms,counts= rand2sym(R,c, sym)
    return ''.join(_syms)

def probability_plot(possible_output, n):
    Outcomes = np.random.randint(possible_output, size = n)
    Count = np.zeros((possible_output,n+1))
    Prob = np.zeros((possible_output,n+1))
    #Counting the occurance of each event
    for i in range(1,n+1):
        Count[:,i] = Count[:,i-1]
        Count[Outcomes[i-1],i]+=1

    # plot the empirical values
    fig, ax = plt.subplots()
    for i in range(possible_output):
        Prob = Count[i,1:]/np.arange(1,n+1)
        ax = plt.plot(np.arange(1, n + 1), Prob, linewidth=2.0, label='Face '+str(i+1))
    
    plt.plot(range(0, n), [1 / possible_output] * n, 'k', linewidth=3.0, label='Theoretical probability')
    plt.title(f"Empirical and theoretical probabilities of the {possible_output} faces")
    plt.xlabel('Number of Iterations')
    plt.ylabel('Probability')
    plt.xlim([1, n])
    plt.ylim([0, 1])
    plt.legend()
    st.pyplot(fig)


st.sidebar.title("Control Panel")
main_control = st.sidebar.selectbox("Directory", options=["Main", "Probability", "Statistics", "Machine Learning"])

if main_control == "Main":
    st.title("Probability, Statistics and Machine learning")
    st.write("This website is for sharing theoretical knowledge, ideas, codes and results")

elif main_control == "Probability":
    st.title("""Experiments about Probability.""")
    selected = st.selectbox("""The theoretical vs simulated situation 
    in tossing a coin or rolling a dice""", ["Select","Coin", "Dice"])
    if selected == "Coin":
        trials_number = st.slider("n", min_value=1, max_value=1000, step=3)
        probability_plot(2, trials_number)
    elif selected == "Dice":
        trials_number = st.slider("n", min_value=1, max_value=1000, step=3)
        probability_plot(6, trials_number)
    else:
        pass
    selected_LongTermFreq = st.selectbox("""This is a stimulation of 4 colors with fixed probability,
    {1:0.1, 2:0.2, 3:0.3, 4:0.4} with color violet, green, orange, blue respectively""", ["Select", "Stimulation"])
    if selected_LongTermFreq == "Stimulation":
        N_Stimulation = st.slider("n", min_value=0, max_value=5000, step=1)
        st.markdown(probability_stimulation(N_Stimulation), unsafe_allow_html=True)
        
elif main_control == "Machine Learning":
    st.title("Algorithms in Machine Learning")
    st.text("This page is for machine learning")
    selected_ml = st.selectbox("""Different machine learning algorithms""", 
        ["Select", "Bivariate Gaussian", "Perceptron", "Support Vector Machine"])
    if selected_ml == "Support Vector Machine":
        selected_data_set_SVM = st.selectbox("""Please select a specific data set""", ["Select","1", "2", "3", "4", "5"])
        if selected_data_set_SVM == "1":
            data_SVM = get_data_SVM(1)
            kernel_function = selected_kernel_function = st.selectbox("Select", ["quadratic", "rbf"])
            # st.dataframe(data_SVM)
            c = st.slider("c value", min_value=0.1, max_value=100.)
            s = st.slider("s value", min_value = 0.01, max_value=20.)
            learn_and_display_SVM(data_SVM, kernel_function, c, s)
        elif selected_data_set_SVM == "2":
            data_SVM = get_data_SVM(2)
            kernel_function = selected_kernel_function = st.selectbox("Select", ["quadratic", "rbf"])
            # st.dataframe(data_SVM)
            c = st.slider("c value", min_value=0.1, max_value=100.)
            s = st.slider("s value", min_value = 0.01, max_value=20.)
            learn_and_display_SVM(data_SVM, kernel_function, c, s)
        elif selected_data_set_SVM == "3":
            data_SVM = get_data_SVM(3)
            kernel_function = selected_kernel_function = st.selectbox("Select", ["quadratic", "rbf"])
            # st.dataframe(data_SVM)
            c = st.slider("c value", min_value=0.1, max_value=100.)
            s = st.slider("s value", min_value = 0.01, max_value=20.)
            learn_and_display_SVM(data_SVM, kernel_function, c, s)
        elif selected_data_set_SVM == "4":
            data_SVM = get_data_SVM(4)
            kernel_function = selected_kernel_function = st.selectbox("Select", ["quadratic", "rbf"])
            # st.dataframe(data_SVM)
            c = st.slider("c value", min_value=0.1, max_value=100.)
            s = st.slider("s value", min_value = 0.01, max_value=20.)
            learn_and_display_SVM(data_SVM, kernel_function, c, s)
        elif selected_data_set_SVM == "5":
            data_SVM = get_data_SVM(5)
            kernel_function = selected_kernel_function = st.selectbox("Select", ["quadratic", "rbf"])
            # st.dataframe(data_SVM)
            c = st.slider("c value", min_value=0.1, max_value=100.)
            s = st.slider("s value", min_value = 0.01, max_value=20.)
            learn_and_display_SVM(data_SVM, kernel_function, c, s)
        else:
            pass

        