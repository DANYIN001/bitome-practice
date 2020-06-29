# third-party modules
import numpy as np
import pywt


def wavelet_scalogram(data, scales=np.arange(75, 1200, 5), wavelet='cmor2.0-0.8', num_bootstraps=200, fdr=0.05):
	"""

	:param np.ndarray data: Data based on genome position to analyze for periodic patterns
	:param np.ndarray scales: the scales at which to perform a wavelet transform to detect periodicity
	:param str wavelet: the wavelet filter function to use with the wavelet transform
	:param int num_bootstraps: the number of randomized transforms to perform when determining significance via
		bootstrapping. Defaults to 200
	:param float fdr: the false discovery rate at which to control significance for the wavelet scale/position
		transform coefficients
	:return np.ndarray binary_significance_matrix: a matrix the same shape as the wavelet scalogram, with 1 entries
		indicating significance of the transform at that scale/position, and 0s insignificance.
	"""

	coeffs = wavelet_transform(data, scales, wavelet)
	rand_scalograms = random_wavelet_transforms(data, scales, wavelet, num_bootstraps)

	# calculate p-values for each entry in the original coefficient scalogram (matrix)
	p_values = np.empty(coeffs.shape)
	for (scale_index, pos_index), coeff in np.ndenumerate(coeffs):
		rand_coeffs_at_index = rand_scalograms[:, scale_index, pos_index]
		p_value = np.mean(abs(rand_coeffs_at_index) > abs(coeff))
		p_values[scale_index, pos_index] = p_value

	significance_threshold = fdr_cutoff(p_values, fdr)

	binary_significance_matrix = p_values < significance_threshold

	return binary_significance_matrix


def wavelet_transform(data, scales, wavelet):
	"""
	Performs the wavelet transform of the data at the given scales, using the provided wavelet.
		Implemented with the PyWavelets package. ONLY RETURNS TRANFORM COEFFICIENTS

	:param np.ndarray data: a 1-D array of raw genomic data, by position
	:param np.ndarray scales: the length (in genome position) scales at which the wavelet transform should be performed
	:param str wavelet: the wavelet (filter) to use in the wavelet transform.
	:return np.ndarray coeffs: the coefficients of the transform at each scale/data position entry
	"""

	# perform the wavelet transform, storing both outputs; return just the coefficients
	coeffs, freqs = pywt.cwt(data, scales, wavelet)
	return coeffs


def random_wavelet_transforms(data, scales, wavelet, num_iterations):
	"""
	Performs wavelet transform on randomized versions of the position-based data for the purpose of bootstrapping,
		returning a 3-D array of all the random coefficients.
		Dimension 1: the iteration index
		Dimension 2: wavelet scales
		Dimension 3: genome positions

	:param np.ndarray data: the initial data on which a wavelet transform was performed
	:param np.ndarray scales: the length (in genome position) scales at which the wavelet transform should be performed
	:param str wavelet: the wavelet (filter) to use in the wavelet transform
	:param int num_iterations: the number of random shuffles of the data on which to re-run the wavelet transform
	:return np.ndarray rand_scalograms: a 3-D array of wavelet scalograms based on randomized source data
	"""

	rand_scalograms = np.empty([num_iterations, len(scales), len(data)])

	for i in range(num_iterations):

		# randomize a copy of the data
		rand_data = np.copy(data)
		np.random.shuffle(rand_data)

		# compute the wavelet transform, add the scalogram to the overall 3-D array
		rand_coeffs = wavelet_transform(rand_data, scales, wavelet)
		rand_scalograms[i] = rand_coeffs

	return rand_scalograms


def fdr_cutoff(p_values, fdr):
	"""

	Given p-values from multiple tests and a desired false discovery rate (FDR), performs the Benjamini-
		Hochberg procedure to adjust the p-values to control FDR. Returns the cutoff threshold for p-value
		significance.

	:param np.ndarray p_values: an array of p-values from multiple testing that need correction to control FDR
	:param float fdr: the false discovery rate (FDR) to enforce via p-value adjustment
	:return float cutoff: the p-value cutoff under which to consider p-values significant, with FDR < alpha
	"""

	# implement my own Benjamini-Hochberg correction since module I tried to use didn't work right
	sorted_p_vals_flat = np.sort(p_values, axis=None)
	m = len(sorted_p_vals_flat)
	p_k = (np.arange(m) + 1) * (fdr / m)
	p_k_tests = sorted_p_vals_flat <= p_k
	stop_index = np.where(p_k_tests == False)[0][0]
	cutoff = sorted_p_vals_flat[stop_index]
	return cutoff
