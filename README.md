# ml-learning
Machine Learning coding practice

## Environment Setup
Due to the use of plotting, most of the code will not run inside a container but directly on the host environment. To setup the host environment, install Python 3, and install the packages below
```
pip3 install --upgrade pip numpy scipy matplotlib scikit-learn pandas
```

## Results

###L4 mini project performance comparison
#### For 10,000 generated terrain data
Note: All non-custom algorithms refer to scikit-learn library

|          | Naive Bayes | Custom KNN | KNN       | Custom AdaBoost | AdaBoost  | Custom Random Forest | Random Forest |
| -------- | ----------- | ---------- | --------- | --------------- | --------- | -------------------- | ------------- |
| Accuracy | 91.36%      | 96.16%     | 96.16%    | 96.12%          | 96.28%    | 96.08%               | 96.32%        |
| Time     | 0.009 sec   | 37.041 sec | 0.017 sec | 6.875 sec       | 0.468 sec | 2.754 sec            | 0.587 sec     |

#### For 100,000 generated terrain data
Note: All non-custom algorithms refer to scikit-learn library

|          | Naive Bayes | Custom KNN | KNN       | Custom AdaBoost | AdaBoost  | Custom Random Forest | Random Forest |
| -------- | ----------- | ---------- | --------- | --------------- | --------- | -------------------- | ------------- |
| Accuracy | 90.97%      |    N/A     | 95.61%    | 96.00%          | 96.17%    | 96.20%               | 96.24%        |
| Time     | 0.067 sec   |    N/A     | 0.162 sec | 59.849 sec      | 3.421 sec | 30.937 sec           | 5.913 sec     |

Navie Bayes is the fastest algorithm, yet accuracy suffers. Random Forest gives the best accuracy, but it is the slowest amongst all the algorithms compared here. On the otherhand, KNN seems to be a great compromise between both worlds. It gives a reasonable good accuracy at a significantly faster speed. For data points of 100,000, it still managed to run in sub-second.

Conclusion: for baseline set up, Navie Bayes seems to be a good one, since we can get a fairly good accuracy very fast. KNN seems to be a good starting point for accuracy improvement. If we need to write custom classifier, then random forest seems to run well.


## Resources/References
### UD120 project
- https://github.com/udacity/ud120-projects
### KNN
- https://www.analyticsvidhya.com/blog/2018/03/introduction-k-neighbours-algorithm-clustering/
- https://medium.com/@adi.bronshtein/a-quick-introduction-to-k-nearest-neighbors-algorithm-62214cea29c7
### Adaboost samples
- https://gist.github.com/tristanwietsma/5486024
- https://github.com/jaimeps/adaboost-implementation
### Random Forest
- https://towardsdatascience.com/random-forest-in-python-24d0893d51c0
- https://machinelearningmastery.com/implement-random-forest-scratch-python/
- https://www.datascience.com/resources/notebooks/random-forest-intro
