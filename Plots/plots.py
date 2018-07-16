import matplotlib.pyplot as plt
import csv

reward_cnn = []
loss_cnn = []
timesteps_cnn = []
reward_lstm = []
loss_lstm = []
timesteps_lstm = []
reward_cpu = []
loss_cpu = []
timesteps_cpu = []


with open('cnn_plotData.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    next(plots)
    for row in plots:
        reward_cnn.append(float(row[0]))
        loss_cnn.append(float(row[1]))
        timesteps_cnn.append(float(row[2]))


with open('lstm_plotdata.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    next(plots)
    for row in plots:
        reward_lstm.append(float(row[0]))
        loss_lstm.append(float(row[1]))
        timesteps_lstm.append(float(row[2]))

with open('cpu_ppo1_data_3M.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    next(plots)
    for row in plots:
        reward_cpu.append(float(row[0]))
        loss_cpu.append(float(row[1]))
        timesteps_cpu.append(float(row[2]))


#PLOT REWARD AND LOSS FOR CNN
reward_norm = [float(i)/max(reward_cnn) for i in reward_cnn]
loss_norm = [float(i)/max(loss_cnn) for i in loss_cnn]

plt.plot(timesteps_cnn,reward_norm, label='Reward')
plt.plot(timesteps_cnn,loss_norm, label='Loss')
plt.xlabel('Timesteps')
plt.ylabel('Reward and Loss')
plt.title('CNN Reward & LOSS plot')
plt.legend()
#plt.show()
plt.savefig("Reward_LOSS_CNN.png")
plt.gcf().clear()

#PLOT REWARD AND LOSS FOR LSTM
reward_norm = [float(i)/max(reward_lstm) for i in reward_lstm]
loss_norm = [float(i)/max(loss_lstm) for i in loss_lstm]

plt.plot(timesteps_lstm,reward_norm, label='Reward')
plt.plot(timesteps_lstm,loss_norm, label='Loss')
plt.xlabel('Timesteps')
plt.ylabel('Reward and Loss')
plt.title('LSTM Reward & LOSS plot')
plt.legend()
#plt.show()
plt.savefig("Reward_LOSS_LSTM.png")
plt.gcf().clear()


#PLOT REWARD AND LOSS FOR CPU - ppo1
reward_norm = [float(i)/max(reward_cpu) for i in reward_cpu]
loss_norm = [float(i)/max(loss_cpu) for i in loss_cpu]

plt.plot(timesteps_cpu,reward_norm, label='Reward')
plt.plot(timesteps_cpu,loss_norm, label='Loss')
plt.xlabel('Timesteps')
plt.ylabel('Reward and Loss')
plt.title('CPU Reward & LOSS plot')
plt.legend()
#plt.show()
plt.savefig("Reward_LOSS_CPU.png")
plt.gcf().clear()







#PLOT REWARD AND LOSS FOR CPU - ppo1 - clip = 0.4
reward_cpu = []
loss_cpu = []
timesteps_cpu = []

with open('clip0.4_ppo1_data.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    next(plots)
    for row in plots:
        reward_cpu.append(float(row[0]))
        loss_cpu.append(float(row[1]))
        timesteps_cpu.append(float(row[2]))

#reward_norm = [float(i)/max(reward_cpu) for i in reward_cpu]
#loss_norm = [float(i)/max(loss_cpu) for i in loss_cpu]

plt.plot(timesteps_cpu,reward_cpu, label='Reward')
plt.plot(timesteps_cpu,loss_cpu, label='Loss')
plt.xlabel('Timesteps')
plt.ylabel('Reward and Loss')
plt.title('Clip 0.4 Reward & LOSS plot - Non Normalized')
plt.legend()
#plt.show()
plt.savefig("Reward_LOSS_CPU_Clip_0.4.png")
plt.gcf().clear()




#PLOT REWARD CNN and LSTM
plt.plot(timesteps_cnn,reward_cnn, label='Reward CNN')
plt.plot(timesteps_lstm,reward_lstm, label='Reward LSTM')
plt.xlabel('Timesteps')
plt.ylabel('Reward and Loss')
plt.title('CNN VS LSTM - Reward')
plt.legend()
#plt.show()
plt.savefig("Reward_CNN_LSTM.png")
plt.gcf().clear()


#PLOT LOSS CNN and LSTM
plt.plot(timesteps_cnn,loss_cnn, label='loss CNN')
plt.plot(timesteps_lstm,loss_lstm, label='Loss LSTM')
plt.xlabel('Timesteps')
plt.ylabel('Loss')
plt.title('CNN VS LSTM - LOSS')
plt.legend()
#plt.show()
plt.savefig("LOSS_CNN_LSTM.png")
plt.gcf().clear()



