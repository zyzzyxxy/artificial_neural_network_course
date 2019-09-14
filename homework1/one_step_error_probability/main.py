import numpy as np

global N
N = 120
stored_pattern_size = 100
stored_pattern_sizes = [12, 24, 48, 70, 100, 120]
results=[[47/10^5,1102/10^5, 5516/10^5, 9430/10^5, 13573/10^5] ]
error_counter = 0


def make_random_pattern(n):
    pattern = np.random.random((n, 1))
    counter = 0

    for bit in pattern:
        if bit >= .5:
            pattern[counter] = 1
        else:
            pattern[counter] = -1
        counter += 1
    pattern = pattern.astype(int)
    return pattern


def calc_one_step_error():
    pass


def make_m_random_nbit_patterns(m, n):
    result = []
    for i in range(m):
        result.append(make_random_pattern(n))
    return result


def calc_weight_matrix_hebbsrule(patterns):
    #weight_matrix = np.zeros(shape=(N, N))
    weight_matrix2 = np.zeros(shape=(N, N))
    # Solution 1
    '''
    for i in range(N):
        for j in range(N):
            for p in patterns:
                if i != j:
                    weight_matrix[i][j] += p[i] * p[j]
    '''
    # Solution 2
    for p in patterns:
        weight_matrix2 += np.matmul(p, p.T)
    for i in range(N):
        weight_matrix2[i][i] = 0

    #weight_matrix = weight_matrix / N
    weight_matrix2 = weight_matrix2 / N

    # print 'w1: \n',weight_matrix,'\nw2: \n', weight_matrix2
    return weight_matrix2


def sgn_random_neuron(pattern, neuron_nbr, weight_matrix):
    result = 0

    for j in range(N):
        result += weight_matrix[neuron_nbr][j] * pattern[j]
    if result < 0:
        result = -1
    else:
        result = 1
    return result


def iteration():
    patterns = make_m_random_nbit_patterns(stored_pattern_size, N)
    weight_matrix = calc_weight_matrix_hebbsrule(patterns)
    # Choose random pattern
    choosen_pattern = patterns[np.random.randint(0, stored_pattern_size)]

    # run random neuron through network
    neuron_number = np.random.randint(0, N)
    updated_neuron = sgn_random_neuron(choosen_pattern, neuron_number, weight_matrix)

    if choosen_pattern[neuron_number] != updated_neuron:
        return 1
    else:
        return 0


for i in range(100000):
    error_counter += iteration()

print 'N: ', N
print 'P: ', stored_pattern_size
print 'Errors in 100000 tries: ', error_counter
