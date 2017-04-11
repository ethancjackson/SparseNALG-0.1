import numpy as np
from scipy import sparse
from lxml.etree import ElementTree, Element, SubElement, tostring


def matrixToGenomeXML(matrix, filename, thresh=0.0001):
    # assert type(matrix) == np.ndarray
    assert matrix.shape[0] == matrix.shape[1], "Matrix must be square."
    if not filename.endswith('.xml'):
        filename += '.xml'

    s_matrix = sparse.coo_matrix(abs(matrix) > thresh)
    src_neuron_set = set(np.unique(s_matrix.row))  # unique redundant
    trg_neuron_set = set(np.unique(s_matrix.col))

    # hidden neurons appear as both src and trg in at least one connection
    hidden_neurons = src_neuron_set.intersection(trg_neuron_set)
    # input neurons never appear as trg
    input_neurons = src_neuron_set.difference(trg_neuron_set)
    # output neurons never appear as src
    output_neurons = trg_neuron_set.difference(src_neuron_set)

    root = Element('genome')
    root.set('id', '0')
    root.set('species-id', '0')
    root.set('age', '1')
    root.set('fitness', '1.0')
    neurons = SubElement(root, 'neurons')

    for n_id in range(matrix.shape[0]):
        if n_id in hidden_neurons:
            SubElement(neurons, 'neuron',
                       {'id': str(n_id), 'type': 'hid', 'activationFunction': 'BipolarSigmoid', 'layer': '1'})
        elif n_id in input_neurons:
            SubElement(neurons, 'neuron',
                       {'id': str(n_id), 'type': 'in', 'activationFunction': 'BipolarSigmoid', 'layer': '0'})
        elif n_id in output_neurons:
            SubElement(neurons, 'neuron',
                       {'id': str(n_id), 'type': 'out', 'activationFunction': 'BipolarSigmoid', 'layer': '2'})

    connections = SubElement(root, 'connections')
    edge_id = matrix.shape[0]
    for src in range(matrix.shape[0]):
        for trg in range(matrix.shape[0]):
            # Create connection element
            if abs(matrix[src, trg]) > thresh:
                weight = matrix[src, trg]
                SubElement(connections, 'connection',
                           {'innov-id': str(edge_id), 'src-id': str(src), 'tgt-id': str(trg), 'weight': str(weight)})
                edge_id += 1
    tree = ElementTree(root)
    tree.write(filename, pretty_print=True)
