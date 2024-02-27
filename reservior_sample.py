import random
from collections import defaultdict


def reservoir_sampling(stream, k):
    """ This function performs reservoir sampling on the input stream.
        It returns a sample size of k from the stream.
    """
    # Initialize a reservoir with the first k items from the stream
    reservoir = []
    for i, item in enumerate(stream):
        if i < k:
            reservoir.append(item)
        else:
            # Randomly replace elements in the reservoir with a decreasing probability.
            # Choose an integer between 0 to i (both inclusive)
            j = random.randint(0, i)
            if j < k:
                reservoir[j] = item

    return reservoir


def reservoir_sampling_max(stream_items, k):
    """
    Reservoir sampling function to select items with maximum occurrence in a stream.

    :param stream_items: A list of tuples, where each tuple consists of an item and its occurrence.
    :param k: The number of items to sample.
    :return: A list containing the most frequently occurring items.
    """
    if k <= 0:
        return []

    # Initialize reservoir and occurrence count
    reservoir = []
    occurrences = defaultdict(int)

    # Process stream items
    for index, (item, count) in enumerate(stream_items):
        if len(reservoir) < k:
            # Fill the reservoir with initial items
            occurrences[item] += count
            reservoir.append((item, occurrences[item]))
        else:
            occurrences[item] += count
            # Check if the current item has more occurrences than the least in the reservoir
            current_min = min(reservoir, key=lambda x: x[1])
            if occurrences[item] > current_min[1]:
                # Replace the least frequent item in reservoir with the current item
                reservoir.remove(current_min)
                reservoir.append((item, occurrences[item]))

    # Sort the reservoir based on occurrence to return the most frequent items
    reservoir.sort(key=lambda x: x[1], reverse=True)

    # Just return the items without their occurrence count
    return [item for item, _ in reservoir]
class ReservoirSamplingMaxOccurrence:
    def __init__(self, k):
        self.k = k  # Reservoir size
        self.counter = {}  # Counter for occurrences
        self.reservoir = []  # Reservoir for holding the samples
    def process_item(self, item):
        # Record occurrence
        self.counter[item] = self.counter.get(item, 0) + 1
        # If the reservoir isn't full, add the item directly
        if len(self.reservoir) < self.k:
            self.reservoir.append(item)
        else:
            # Consider replacing an item with lower occurrence
            potential_replacements = [i for i, stored_item in enumerate(self.reservoir)
                                      if self.counter[stored_item] < self.counter[item]]
            if potential_replacements:
                replacement_idx = random.choice(potential_replacements)
                self.reservoir[replacement_idx] = item
    def get_samples(self):
        return self.reservoir

# Example usage:
if __name__ == "__main__":
    # Let's say we have a stream of numbers from 1 to 100, and we want
    # to get a random sample of 10.
    # stream = range(1, 101)
    # k = 10
    # sample = reservoir_sampling(stream, k)
    # print("Random sample of size", k, "from the stream:", sample)
    #
    # # Usage
    # reservoir_size = 10
    # sampler = ReservoirSamplingMaxOccurrence(reservoir_size)
    # # Example: Process a stream of items
    # stream = ['a', 'b', 'a', 'c', 'a', 'b', 'd', 'a', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p']
    # for item in stream:
    #     sampler.process_item(item)
    # # Get the samples from the reservoir
    # samples = sampler.get_samples()
    # print("Reservoir:", samples)

    stream = [('apple', 3), ('banana', 2), ('orange', 5), ('grape', 1), ('peach', 4)]
    k = 3
    print(reservoir_sampling_max(stream, k))