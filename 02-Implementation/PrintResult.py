import numpy as np
from prettytable import PrettyTable

def Print_Result(net_history, test):
    
    values = []
    
    for i in range(len(net_history)):
        
        acc = net_history[i].history['accuracy']
        min_acc = min(acc)
        index_min_acc = acc.index(min_acc)
        max_acc = max(acc)
        index_max_acc = acc.index(max_acc)
        
        
        val_acc = net_history[i].history['val_accuracy']
        min_val_acc = min(val_acc)
        index_min_val_acc = val_acc.index(min_val_acc)
        max_val_acc = max(val_acc)
        index_max_val_acc = val_acc.index(max_val_acc)
        if i%2 == 0: j = 3
        else: j = 4
        values.append([i+1, j, 100,
                       index_min_acc+1,"{:.2f}".format(min_acc*100),
                       index_min_val_acc+1,"{:.2f}".format(min_val_acc*100),
                       index_max_acc+1,"{:.2f}".format(max_acc*100),
                       index_max_val_acc+1,"{:.2f}".format(max_val_acc*100),
                       "{:.2f}".format(np.mean(val_acc)*100)])
        
    x = PrettyTable()
    x.field_names = ['Case', 'Num Hidden Layers', 'Batch Size',
                 'Min Train Acc Epoch',
                 'Min Train Acc %',
                 'Min Val Acc Epoch',
                 'Min Val Acc %',
                 'Max Train Acc Epoch',
                 'Max Train Acc %',
                 'Max Val Acc Epoch',
                 'Max Val Acc %',
                 'Overall Performance Val Acc %']

    x.add_rows(values)
    
    y = PrettyTable()
    y.field_names = ['Case', 'Test Loss', 'Test Acc']
    
    for i in range(len(test)):
        t_loss = "{:.2f}".format(test[i][0]*100)
        t_acc = "{:.2f}".format(test[i][1]*100)
        y.add_row([i+1, t_loss, t_acc])
   
    print(x)
    print(y)


        
