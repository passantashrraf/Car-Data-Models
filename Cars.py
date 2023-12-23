import pandas
import matplotlib.pyplot as plot


def optimalFunction(errors):
    min_value = min(errors)
    min_index = errors.index(min_value)
    return min_index


def gradientDescent(thetas, data, target, alpha, trainset, loops):
    thetaOrder = 0
    for i in range(loops):
        meanError = meanSquareError(trainset, target, data, thetas)
        meanErrors.append(meanError)
        allThetas.append(thetas)
        if thetaOrder == 0:
            thetas[thetaOrder] = thetas[thetaOrder] - alpha * (1 / trainset) * meanErrorDerivation(thetaOrder, thetas,
                                                                                                   data, trainset,
                                                                                                   target,
                                                                                                   'dummyfeature')
            thetaOrder += 1
        for feature in data:
            if thetaOrder >= 5:
                break
            thetas[thetaOrder] = thetas[thetaOrder] - alpha * (1 / trainset) * meanErrorDerivation(thetaOrder, thetas,
                                                                                                   data, trainset,
                                                                                                   target, feature)
            thetaOrder += 1


def meanErrorDerivation(thetaOrder, thetas, data, trainset, target, feature):
    sum = 0
    for rownum in range(trainset):
        predicted = hypothesisFunction(thetas, data, rownum)
        actual = target[rownum]

        if thetaOrder == 0:
            sum += predicted - actual
        else:
            sum += (predicted - actual) * data[feature][rownum]
    return sum


def meanSquareError(trainset, target, data, thetas):  # return sum of errors based on a theta group
    errorsum = 0
    for rownum in range(trainset):
        predicted = hypothesisFunction(thetas, data, rownum)
        actual = target[rownum]
        errorsum += (predicted - actual) ** 2
    return errorsum/(2*trainset)


def hypothesisFunction(thetas, data, rownum):
    sum = thetas[0]
    thetaOrder = 1
    for f in data:
        sum += thetas[thetaOrder] * data[f][rownum]
        thetaOrder += 1
    return sum


# 1 read data from csv file
carsData = pandas.read_csv('car_data.csv')
linearfeatures = ['horsepower', 'curbweight', 'enginesize', 'boreratio', 'stroke']
features = ['name', 'drivewheels', 'horsepower', 'curbweight', 'enginesize', 'boreratio', 'stroke']

# 2 scatter plot 7 features
for i in features:
    carsData.plot(kind='scatter', x=i, y='price')
    plot.show()
# 3 split data into training and testing set
# put the numbers to array
horsepower = carsData['horsepower'].tolist()
curbweight = carsData['curbweight'].tolist()
enginesize = carsData['enginesize'].tolist()
boreratio = carsData['boreratio'].tolist()
stroke = carsData['stroke'].tolist()
price = carsData['price'].tolist()
# split
trainSet = 130
horsepower_train = horsepower[0:trainSet]
curbweight_train = curbweight[0:trainSet]
enginesize_train = enginesize[0:trainSet]
boreratio_train = boreratio[0:trainSet]
stroke_train = stroke[0:trainSet]
price_train = price[0:trainSet]
# features_train = np.array([[horsepower_train, curbweight_train, enginesize_train,boreratio_train,stroke_train]])
features_train = {'horsepower': horsepower_train,
                  'curbweight': curbweight_train,
                  'enginesize': enginesize_train,
                  'boreratio': boreratio_train,
                  'stroke': stroke_train
                  }
testset = 75
horsepower_test = horsepower[trainSet:]
curbweight_test = curbweight[trainSet:]
enginesize_test = enginesize[trainSet:]
boreratio_test = boreratio[trainSet:]
stroke_test = stroke[trainSet:]
price_test = price[trainSet:]
features_test = {'horsepower': horsepower_test,
                 'curbweight': curbweight_test,
                 'enginesize': enginesize_test,
                 'boreratio': boreratio_test,
                 'stroke': stroke_test
                 }
# 4 implement linear regression
# declarations of hypothesis
alpha = 0.9
iterations = 9900
thetas = [0.001, 0.001, 0.001, 0.001, 0.001, 0.001]
# data = {'1': [1, 2, 3],
#         '2': [6, 2, 3],
#         '3': [1, 2, 3],
#         '4': [1, 2, 3],
#         '5': [1, 2, 3]}
# target = [5, 7, 8]
meanErrors = []
allThetas = []
gradientDescent(thetas, features_train, price_train, alpha, trainSet, iterations)
index = optimalFunction(meanErrors)
print(index)
print('///////////////////////////////////')
# 5 print the hypothesis function parameters
print("optimal parameters",allThetas[index])
print('///////////////////////////////////')
# 6 mean square errors in every iteration are printed here
print("optimal mean error:",meanErrors[index])
print('///////////////////////////////////')
# 7
iters=[]
for i in range (iterations):
    iters.append(i)
y = meanErrors
x = iters
plot.plot(x, y)
plot.show()
# 8 testing sets
accuracy = meanSquareError(testset, price_test, features_test, allThetas[index])
print('mean error of tested set:', accuracy)
