import matplotlib.pyplot as plt


def Plot_History(net_history):
    
    fig, axs = plt.subplots(3,2)
    fig.suptitle('Models Acc and val_Acc')
    
    for i in range(len(net_history)):
        if i%2 == 0 : j =0
        else: j = 1 
        axs[int(i/2),j].title.set_text('Model Case ' + str(i+1))
        axs[int(i/2),j].set_ylabel('Accuracy')
        axs[int(i/2),j].grid()
        axs[int(i/2),j].plot(net_history[i].history['accuracy'], '-o')
        axs[int(i/2),j].plot(net_history[i].history['val_accuracy'], '-o')
        axs[int(i/2),j].legend(['acc', 'val_acc'])
    
    plt.show()
    