import math
from confusionMatrix import computeAccuracy
from confusionMatrix import printConfMatrix
#from plot_helper import Result
#from graph_Plotter import resultWriter

class LogisticRegression:
    def __init__(self):
        self.tp = 0.0
        self.tn = 0.0
        self.fp = 0.0
        self.fn = 0.0
        self.tpr = 0.0
        self.fpr = 0.0
        self.accuracy = 0.0
        #self.result_list = []

    def predict(self, row, weights):
        #initialize with the y-intercept
        #print row
        predicted_label = weights[0]
        for i in range(len(row)-1):
            #print "Row = ",row
            #print "Row val = ", row[i]
            #raw_input()
            #print "Computing ....",predicted_label, weights[i+1], row[i]
            predicted_label += weights[i+1] * row[i]

        if predicted_label >= 0:
            exponent = math.exp(-predicted_label)
            return 1 / (1 +exponent)
        else:
            exponent = math.exp(predicted_label)
            return exponent / (1 + exponent)

    def update_weights(self, weights, predicted_values, data, etta):
        gradient = [0.0] * len(weights)
        for i in range(len(data)):
            error = data[i][-1] - predicted_values[i]
            #print "error....",error
            for j in range(1, len(weights)):
                gradient[j] = error * data[i][j-1]
            weights[0] += error * etta
            for j in range(len(weights)):
                weights[j] += gradient[j]

    def compute_logistic_regression(self, train_data, test_data):
        etta = 0.9
        iterations = 10
        while etta >=0.1:
            x_list = []
            y_list = []
            iterations = 10
            print "Computing predictions for etta= ",etta, "....."
            while iterations <=100:
                weights = [0.0] * (len(train_data[0]))
                #print "Training weights..."
                for i in range(iterations):
                    predictions = self.predict_labels(train_data, weights)
                    self.update_weights(weights, predictions, train_data, etta)
                #print "Testing!!! "
                accuracy = self.test_LR(test_data, weights)
                #x_list.append(iterations)
                #y_list.append(accuracy)
                iterations +=10
            #result = Result(x_list, y_list, etta)
            #self.result_list.append(result)
            print "Etta == ", etta
            self.print_results()
            #raw_input()
            self.initialize_args()
            etta = etta **2
        #rw = resultWriter("LR")
        #rw.plot_curve_LR(self.result_list)



    def print_results(self):
        tp = self.tp / 10
        tn = self.tn / 10
        fp = self.fp / 10
        fn = self.fn / 10
        accuracy = self.accuracy / 10
        tpr = self.tpr / 10
        fpr = self.fpr / 10
        print "Average accuracy = ",accuracy
        print "Average TPR = ", tpr
        print "Average FPR = ", fpr
        printConfMatrix(tp, fp, tn, fn)

    def initialize_args(self):
        self.tp = 0.0
        self.tn = 0.0
        self.fp = 0.0
        self.fn = 0.0
        self.tpr = 0.0
        self.fpr = 0.0
        self.accuracy = 0.0


    def predict_labels(self, data, weights):
        predictions = []
        for r in range(len(data)):
            predictions.append(self.predict(data[r], weights))
        return predictions

    def test_LR(self, test_data, weights):
        correct = 0
        #print "Weights:"
        #print weights
        predictions = self.predict_labels(test_data, weights)
        for i in range(len(predictions)):
            predictions[i] = round(predictions[i])
        result_metric = computeAccuracy(test_data, predictions)
        self.update_results(result_metric)
        return result_metric.accuracy

    def update_results(self, result_metric):
        self.tp += result_metric.tp
        self.tn += result_metric.tn
        self.fp += result_metric.fp
        self.fn += result_metric.fn
        self.tpr += result_metric.tpr
        self.fpr += result_metric.fpr
        self.accuracy += result_metric.accuracy


