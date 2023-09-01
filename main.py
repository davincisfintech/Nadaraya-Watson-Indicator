
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from pprint import pprint

df=pd.read_csv('close.csv')
close=list(df['Close'])[-500:]


def gaussian_kernel(x, h):
    return np.exp(-(x ** 2) / (h * h * 2))

def nadaraya_watson_kernel_regression(x, y, x_eval, bandwidth):
    weights = gaussian_kernel(x - x_eval, bandwidth)
    weights /= np.sum(weights)
    return np.dot(y, weights)

def nadaraya_watson_envelope(source, bandwidth, multiplier):
    num_data_points = len(source)
    upper_envelope = []
    lower_envelope = []

    for i in range(num_data_points):
        lower_bound = max(0, i - bandwidth)
        upper_bound = min(num_data_points, i + bandwidth + 1)
        local_data_points = source[lower_bound:upper_bound]

        regression_value = nadaraya_watson_kernel_regression(
            np.arange(lower_bound, upper_bound), local_data_points, i, bandwidth
        )

        envelope_value = regression_value + multiplier * np.std(local_data_points)
        upper_envelope.append(envelope_value)

        envelope_value = regression_value - multiplier * np.std(local_data_points)
        lower_envelope.append(envelope_value)

    return upper_envelope, lower_envelope

close_data =close

bandwidth = 8
multiplier = 3
upper_env, lower_env = nadaraya_watson_envelope(close_data, bandwidth, multiplier)
print(list(df['Date'])[-1])
print(upper_env[-1], lower_env[-1])

plt.plot(close_data, label='Close')
plt.plot(upper_env, label='Upper Envelope')
plt.plot(lower_env, label='Lower Envelope')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Nadaraya-Watson Envelope Indicator')
plt.grid(True)
plt.show()

