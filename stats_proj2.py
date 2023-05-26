import random
import matplotlib.pyplot as plt
import numpy as np
a = 4
b = 5
c = 7
p = b*c
seq_len = 60
strk_len = 2
probs_of_samples = []

def run_sequence(strk, is_sample, samples, seq_len):
	for j in range(seq_len):
	        if random.random() < p/100:
	            print(1)
	            strk += 1
	            if is_sample:
	                samples.append(1)
	                is_sample = False
	        else:
	            print(0)
	            strk = 0
	            if is_sample:
	                samples.append(0)
	                is_sample = False
	        print(f"streak: {strk}")
	        if strk == 2:
	            strk = 1
	            is_sample = True
	        print(f"{is_sample} {samples}")

#----------------------------------------------------------------------------------------------------------------------------

number_of_invalid_samples = 0
for i in range(10000):
    strk = 0
    is_sample = False
    samples = []
    run_sequence(strk, is_sample, samples, seq_len)
    if len(samples) > 0:
        probs_of_samples.append(sum(samples) / len(samples))
    else:
    	number_of_invalid_samples += 1
    print(probs_of_samples)
    print()

while number_of_invalid_samples > 0: #replace non-samples
    strk = 0
    is_sample = False
    samples = []
    run_sequence(strk, is_sample, samples, seq_len)
    if len(samples) > 0:
        probs_of_samples.append(sum(samples) / len(samples))
        number_of_invalid_samples -= 1

 #---------------------------------------------------------------------------------------------------------------------------------

length = len(probs_of_samples) 
sorted_probs = sorted(probs_of_samples)
index_25th = int((25/100) * (length-1))
index_75th = int((75/100) * (length-1))

min_val = min(probs_of_samples)
q1 = sorted_probs[index_25th]
if length % 2 == 0:
    median = (sorted_probs[length//2] + sorted_probs[length//2 - 1]) / 2
else:
    median = sorted_probs[length//2]
q3 = sorted_probs[index_75th]
max_val = max(probs_of_samples)
mu = sum(probs_of_samples) / length
variance = (sum((x - mu) ** 2 for x in probs_of_samples) / length)
sigma = variance ** 0.5
iqr = q3 - q1
range_x = max_val - min_val

for i, prob in enumerate(probs_of_samples):
	print(f"{i+1}:{int(prob*100)*'■'} {prob} ")
print(probs_of_samples)
print(f"min: {min_val} q1: {q1} median: {median} q3: {q3} max: {max_val} mean: {mu} variance: {variance} standard_deviation: {sigma} IQR: {iqr} range: {range_x}")

#-----------------------------------------------------------------------------------------

x = np.arange(length)

# Create a bar plot of the PDF
plt.bar(x, probs_of_samples)

# Set labels and title
plt.xlabel('Sample')
plt.ylabel('Probability')
plt.title('Probability Per Sample')

# Show the plot
plt.show()

#---------------------------------------------------------------------------------------------------

# Generate x-axis values
x = np.linspace(q1 - 3*sigma, q3 + 3*sigma, 1000)

# Calculate the probability density function (PDF)
pdf = 1/(np.sqrt(2*np.pi*variance)) * np.exp(-0.5*((x-mu)/sigma)**2)

# Plot the PDF
plt.plot(x, pdf)

# Find the x-value at the peak of the PDF
peak_x = x[np.argmax(pdf)]

# Label the peak x-value on the plot
plt.annotate(f"Mean: {peak_x:.3f}", xy=(peak_x, np.max(pdf)), xytext=(peak_x, np.max(pdf)*0.8),
             arrowprops=dict(facecolor='black', arrowstyle='->'))

# Set labels and title
plt.xlabel('Values')
plt.ylabel('Probability Density')
plt.title('Probability Density Function')

# Show the plot
plt.show()

#-----------------------------------------------------------------------------------

# Population parameters
population_mean = mu
population_std_dev = sigma

# Sample size and number of samples
sample_size = seq_len
num_samples = 10000

# Simulate sampling distribution
sampling_distribution = []
for _ in range(num_samples):
    sample = np.random.normal(population_mean, population_std_dev, sample_size)
    sample_mean = np.mean(sample)
    sampling_distribution.append(sample_mean)

# Plot the sampling distribution
plt.hist(sampling_distribution, bins=30, edgecolor='black')
plt.xlabel('Sample Mean')
plt.ylabel('Frequency')
plt.title('Sampling Distribution')
plt.show()
