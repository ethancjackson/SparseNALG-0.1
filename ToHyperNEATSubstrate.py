import numpy as np
from scipy import sparse
from lxml.etree import ElementTree, Element, SubElement, tostring


def matrixToSubstrate(matrix, filename, thresh=0.0001):
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

    num_in = int(len(input_neurons))
    num_hid = int(len(hidden_neurons))
    num_out = int(len(output_neurons))

    DX_in = 2 / num_in
    DX_out = 2 / num_out

    hid_row_size = int(np.ceil(np.sqrt(num_hid)))

    DX_hid = 2 / float(hid_row_size)
    DY_hid = 2 / (2 + float(hid_row_size))

    in_y = -1
    out_y = 1

    root = Element('substrate')
    root.set('leo', 'false')
    neuronGroups = SubElement(root, 'neuronGroups')

    start_x = -1
    for n_id in input_neurons:
        SubElement(neuronGroups, 'group',
                   {'id': str(n_id), 'type': 'Input', 'startx': str(start_x), 'endx': str(start_x + DX_in),
                    'starty': str(in_y), 'endy': str(in_y), 'dx': str(num_in), 'dy': str(1)})
        start_x += DX_in

    start_x = -1
    start_y = -1 + DY_hid
    row_counter = 0
    for n_id in hidden_neurons:
        SubElement(neuronGroups, 'group',
                   {'id': str(n_id), 'type': 'Hidden', 'startx': str(start_x), 'endx': str(start_x + DX_hid),
                    'starty': str(start_y), 'endy': str(start_y + DY_hid), 'dx': str(hid_row_size),
                    'dy': str(hid_row_size)})
        start_x += DX_hid
        row_counter += 1
        if row_counter >= hid_row_size:
            # move to a new row
            start_x = -1
            start_y += DY_hid
            row_counter = 0

    start_x = -1
    for n_id in output_neurons:
        SubElement(neuronGroups, 'group',
                   {'id': str(n_id), 'type': 'Output', 'startx': str(start_x), 'endx': str(start_x + DX_out),
                    'starty': str(out_y), 'endy': str(out_y), 'dx': str(num_out), 'dy': str(1)})
        start_x += DX_out

    connections = SubElement(root, 'connections')
    edge_id = len(matrix)
    for src in range(len(matrix)):
        for trg in range(len(matrix)):
            # Create connection element
            if abs(matrix[src, trg]) > thresh:
                SubElement(connections, 'connection',
                           {'src-id': str(src), 'tg-id': str(trg), 'type': 'fully-connected'})
                edge_id += 1
    tree = ElementTree(root)
    tree.write(filename, pretty_print=True)
