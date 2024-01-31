from LoadData import Load_Data
from PlotHistory import Plot_History
from CreateModel import Create_Models
from PrintResult import Print_Result

if __name__ == '__main__':

    X_train, X_test, Y_train, Y_test = Load_Data()
    network_history = []
    test_history = []

    for model in Create_Models():
        
        network_history.append( model.fit(X_train, Y_train, batch_size=100, epochs=15, validation_split=0.2) )
        test_history.append( model.evaluate(X_test, Y_test) )

    Plot_History(network_history)
    Print_Result(network_history, test_history)

    







