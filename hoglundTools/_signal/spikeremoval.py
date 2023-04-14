#Moddified from thomasaarholt
#https://gist.github.com/thomasaarholt/f85ef8360682a256de260d0343d77f83#file-spikeremoval-py

def median_from_neighbors(energy_slice, x, y=""):
	"""Takes the median of the value of the list neighbors"""
	import numpy as np
	return(np.median(find_cell_neighbors(energy_slice, x,y)))

def find_cell_neighbors(data, X, Y, r=1):
	"""Finds value of neighbors of a index in the SI, excluding the centre and any values outside the edges of the SI"""
	adjacent = []

	if Y == "":
		# Line spectrum
		for x in range(X-r, X+r+1):
			if (x == X): # Do not include the spike itself
				pass
			elif (x < 0) or (x >= data.shape[0]): # Do not include indices outside the edges
				pass
			else:
				adjacent.append(data[x])
	else:
		# map
		for x in range(X-r, X+r+1):
			for y in range(Y-r, Y+r+1):
				if (x == X) and (y == Y): # Do not include the spike itself
					pass
				elif (x < 0) or (x >= data.shape[0]): # Do not include indices outside the edges
					pass
				elif (y < 0) or (y >= data.shape[1]): # Do not include indices outside the edges
					pass
				else:
					adjacent.append(data[x,y])
	return(adjacent)

def plot_spike_histogram(diff, diff_after):
	"""Plots a single plot of the histogram of data differential before and after spikes removal"""
	import matplotlib.pyplot as plt
	import numpy as np

	figure = plt.figure()
	hist_before, bins_before = np.histogram(diff, bins="auto")
	hist_after, bins_after = np.histogram(diff_after, bins="auto")
	center_before = (bins_before[:-1] + bins_before[1:]) / 2
	center_after = (bins_after[:-1] + bins_after[1:]) / 2
	width = 5.0 # Width of column

	plt.bar(center_before, hist_before, width=width, color = "red", label='before', log=True)
	plt.bar(center_after, hist_after, width=width, color = "blue", label='after', log=True)

	ax = plt.gca()
	(ymin, ymax) = ax.get_ylim()
	ax.set_ylim([1e-1,ymax])
	plt.legend()
	plt.title("Histogram of signal derivative")
	plt.xlabel("Derivative Magnitude")
	plt.ylabel("Counts")
	return

def remove_spikes(s=None, nMAD = 8, plot_difference=False):
	"""Removes spikes outside nMad Median Absolute Deviations of the median of the differential of the spectrum"""
	from statsmodels.robust import scale
	import numpy as np

	shape = s.data.shape
	if len(shape) == 1:
		# Case single spectrum
		raise AttributeError("Remove spikes requires a Spectrum Image with non-zero navigation dimension")

	elif len(shape) == 2:
		# Case Line profile
		print("Recognised as Line Profile")
		diff = np.diff(s.data, axis=0)
		threshold = nMAD * scale.mad(diff.flatten()) # MAD - nMad is the number of deviations away from the median are included
		print("Gradient threshold is " + str(threshold))

		spike_positions = []

		positive = diff > threshold # Position of any inclining spikes
		(x,e) = np.nonzero(positive) # Get index of spikes
		x += 1 # Position of spike is the array index ahead of the gradient
		for i in range(len(x)):
			spike_positions.append([x[i], e[i]])

		negative = diff < -threshold # Position of any declining spikes
		(x,e) = np.nonzero(negative) # Get index of spikes
		for i in range(len(x)):
			spike_positions.append([x[i], e[i]])

		print("Found " + str(len(spike_positions)) + " spikes!")

		for (x,e) in spike_positions:
			s.data[x,e] = median_from_neighbors(s.data[:,e], x)

		if plot_difference == True:
			diff_after = np.diff(s.data, axis=0)
			plot_spike_histogram(diff, diff_after)

	elif len(shape) == 3:
		# Case EELS Map
		print("Recognised as 2D Spectrum Image")
		diff = np.diff(s.data, axis=1) # Get differential across data
		threshold = nMAD * scale.mad(diff.flatten()) # MAD
		print("Gradient threshold is " + str(threshold))

		positive = diff > threshold # Position of any inclining spikes
		negative = diff < -threshold # Position of any declining spikes

		spike_positions = []

		(x,y,e) = np.nonzero(positive) # Get index of spikes
		y += 1 # Position of spike is the array index ahead of the gradient
		for i in range(len(x)):
			spike_positions.append([x[i], y[i], e[i]])

		(x,y,e) = np.nonzero(negative) # Get index of spikes
		for i in range(len(x)):
			spike_positions.append([x[i], y[i], e[i]])

		print("Found " + str(len(spike_positions)) + " spikes!")

		for (x,y,e) in spike_positions:
			# Spike intensity replaced by median of neighbors
			s.data[x,y,e] = median_from_neighbors(s.data[:,:,e], x,y)

		if plot_difference == True:
			diff_after = np.diff(s.data, axis=1)
			plot_spike_histogram(diff, diff_after)
	else:
		print("The signal shape does not match an expected value (X, S or X, Y, S). Signal data shape is " + str(s.data.shape))

	return s.deepcopy()