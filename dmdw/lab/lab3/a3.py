import math
from typing import List
import matplotlib.pyplot as plt


def make_size_power_of_two(data: List[float]) -> List[float]:
    """
    Haar wavelet transform works only on power of two data
    So, Append zeros to the data until the data is a power of two
    """
    s = len(data)
    req_s = 2**math.ceil(math.log2(s))

    new_data = data.copy()
    for _ in range(s, req_s):
        new_data.append(0)

    return new_data


def transform(data : List[float]) -> List[float]:
    """
    Apply Haar-wavelet transform on the data

    """

    averages = data
    transformed_data = []
    new_averages = []
    while len(averages) > 1:
        details_coeff = []
        for i in range(0, len(averages), 2):
            new_averages.append((averages[i] + averages[i+1]) / 2)
            details_coeff.append((averages[i] - averages[i+1]) / 2)
        transformed_data = details_coeff + transformed_data
        averages = new_averages
        new_averages = []
    transformed_data = averages + transformed_data

    return transformed_data


def apply_threshold(data : List[float], threshold : float) -> List[float]:
    new_data = [d if d>=threshold else 0 for d in data]
    return new_data

def inverse_transform(data):
    """
    Apply inverse Haar-wavelet transform on the data
    """
    
    averages = data[:1]
    details_coeff = data[1:]

    while len(details_coeff) > 0:
        new_averages = []
        for av in averages:
            new_averages.append(av+details_coeff[0])
            new_averages.append(av-details_coeff[0])
            details_coeff = details_coeff[1:]
        averages = new_averages
    return averages



def plot(resized_data, transformed_data, inverse_transformed, threshold=None):
    """
    This function plots the original data, transformed data and
    reconstructed data.
    """
    
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.plot(resized_data, label='original', marker='o', linestyle='dashed')
    ax1.plot(transformed_data, label='transformed', marker='x', linestyle=':')
    ax1.plot(inverse_transformed, label='reconstructed', marker='v', linestyle='-.', alpha=0.6)

    if threshold:
        title = f"Applying Haar Wavelet transform with threshold = {threshold}"
    else:
        title ="Applying Haar Wavelet transform"
    
    plt.title(title)

    plt.legend(loc="upper right")
    plt.show()


def haar_wavelet_transform():
    """
    This function takes input from the user and calls the
    required functions to perform Haar wavelet transform
    """
    
    print("Enter the data that you want to transform:")
    data = [float(x) for x in input().split()]
    
    # making sure that the data is a power of two
    resized_data = make_size_power_of_two(data)

    # applying the haar wavelet transform
    transformed_data = transform(resized_data)
    print("Transformed data:", transformed_data)

    # what is the threshold?
    print("What is the Threshold?")
    threshold = float(input())

    print(f"Applying threshold = {threshold} ...")
    # applying the threshold
    threshold_applied = apply_threshold(transformed_data, threshold)

    # applying the inverse transform
    inverse_transformed = inverse_transform(threshold_applied)
    print("Reconstructed data:", inverse_transformed)

    # plotting the results
    plot(resized_data, transformed_data, inverse_transformed, threshold)
    print("")


if __name__ == "__main__":
    haar_wavelet_transform()

