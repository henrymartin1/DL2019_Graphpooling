from torch_geometric.datasets import Planetoid, CoraFull

for dataset_name in ['Cora', 'PubMed', 'CoraFull']:
    print(dataset_name)
    
    if dataset_name == 'CoraFull':
        dataset = CoraFull(root='/tmp/CoraFull')
    elif dataset_name == 'PubMed':
        dataset = Planetoid(root='/tmp/PubMed', name=dataset_name)
    else:
        dataset = Planetoid(root='/tmp/Cora', name=dataset_name)
        
    print("num classes=", dataset.num_classes)

    data = dataset[0]
    print("num nodes=", data.num_nodes)

    print("num edges=", data.num_edges/2)

    print("num features=", dataset.num_node_features)

