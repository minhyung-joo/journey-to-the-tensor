import math

# Training data is a vector of real numbers and a label
train_data = [
    ([2, 3, 4], 1),
    ([3, 4, 5], 1),
    ([1, 3, 5], 0),
    ([0, 2, 4], 0),
]

# Test data is a vector of real numbers without label
test_data = [
    [0.5, 1, 1.5],
    [5, 6, 7]
]

def calculate_distance_from(v2):
    def calculate_distance_to(v1):
        vector = v1[0]
        if len(vector) != len(v2):
            return -1
    
        squared_sum = 0
        for i in range(len(vector)):
            squared_sum = squared_sum + (vector[i] - v2[i]) ** 2
        
        return math.sqrt(squared_sum)
    
    return calculate_distance_to

# Function for calculating Euclidean distance
def calculate_distance(v1, v2):
    if len(v1) != len(v2):
        return -1
    
    squared_sum = 0
    for i in range(len(v1)):
        squared_sum = squared_sum + (v1[i] - v2[i]) ** 2
    
    return math.sqrt(squared_sum)

# Parameter for how many neighbors to check
k = 3

for data in test_data:
    # Sort the train data based on Euclidean distance from test data
    sorted_data = sorted(train_data, key=calculate_distance_from(data))
    print (sorted_data)
    label_sum = 0
    # Check labels of the first k items in sorted list
    for i in range(k):
        label_sum = label_sum + sorted_data[i][1]
    
    # If the sum is greater than half of k, we classify it as 1
    if label_sum > k / 2:
        print (data, 1)
    else:
        print (data, 0)
