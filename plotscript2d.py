import pickle
import matplotlib.pyplot as plt
import numpy as np

with open("fnnpred2d_plot", "rb") as fp:   #Pickling
    a = pickle.load(fp)

print('test')
with open("rnnpred2d_plot", "rb") as fp:   #Pickling
    b = pickle.load(fp)

with open("lstmpred2d_plot", "rb") as fp:   #Pickling
    c = pickle.load(fp)

with open("gprpred2d_plot", "rb") as fp:   #Pickling
    d = pickle.load(fp)

# open true SOH data values:
with open("true2d_plot.pkl", "rb") as fp:   #Pickling
    e = pickle.load(fp)

# get pred and cycle values for models
fnn = a[0]
rnn = b[0]
lstm = c[0]
gpr = d[0]
cycles_fnn = a[1]
cycles_rnn = b[1]
cycles_lstm = c[1]
cycles_gpr = d[1]



# get pred and cycle values for true SOH
true = e[0]
cycles_true = e[1]


# put data in right order:
cycles_fnn, fnn = zip(*sorted(zip(cycles_fnn, fnn)))
cycles_rnn, rnn = zip(*sorted(zip(cycles_rnn, rnn)))
cycles_lstm, lstm = zip(*sorted(zip(cycles_lstm, lstm)))
cycles_gpr, gpr = zip(*sorted(zip(cycles_gpr, gpr)))



# connected plot
plt.figure(figsize=(7, 5.5))

# multiply by 100 for SOH %
fnn = np.asarray(fnn)*100
rnn = np.asarray(rnn)*100
lstm = np.asarray(lstm)*100
gpr = np.asarray(gpr)*100
true = np.asarray(true)*100

## scatter plot with markers 1
# plt.scatter(cycles_fnn, fnn, label='FNN', marker='.')
# plt.scatter(cycles_rnn, rnn, label='RNN', marker='.', color='purple')
# plt.scatter(cycles_lstm, lstm, label='LSTM', marker='.')
# plt.scatter(cycles_gpr, gpr, label='GPR', marker='.')

# second set of markers 2
plt.scatter(cycles_fnn, fnn, label='FNN', marker='.', s=45)
plt.scatter(cycles_rnn, rnn, label='RNN', marker='*', color='purple', s=35)
plt.scatter(cycles_lstm, lstm, label='LSTM', marker='x', s=30)
plt.scatter(cycles_gpr, gpr, label='GPR', marker='d', s=20)

plt.plot(cycles_true, true, label='True SOH', color='red')
plt.xlabel('Cycle Number')
plt.ylabel('SOH %')
plt.legend()
plt.show()



print('test')