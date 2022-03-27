import torch
import torch.nn.functional as F

################################################################################
#  evaluate the model prediction accuracy
################################################################################
def evaluate_multihead(model, graph, features, labels, mask, task_index):
    #prepare for evaluation
    model.eval()

    with torch.no_grad():
        #compute the predictions
        logits = model(graph, features, task_index)
        predictions = F.log_softmax(logits)

        predictions = predictions[mask]
        labels = torch.LongTensor(labels)[mask]
        _, indices = torch.max(predictions, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


################################################################################
#  evaluate the all neighbors' prediction accuracy
################################################################################
def evaluate_neighbor_acc(model, graph, node_features, node_labels, neighbor_list):
    acc = []

    for i in range(len(neighbor_list)):
        test_mask = id_list_to_mask(neighbor_list[i],2708)
        current_acc = evaluate(model, graph, node_features, node_labels, test_mask)
        acc.append(current_acc)

    return acc


################################################################################
#  find the percentage of neighbor labels that is the target_label
################################################################################
def same_label_percentage(model, graph, node_features, node_labels, neighbor_label_list, target_label):
    same_label_percentage = []

    for i in range(len(neighbor_label_list)):
        current_list = neighbor_label_list[i]
        total_vertex = len(current_list)

        vertex_of_same_label = 0
        for vertex_label in current_list:
            if vertex_label == target_label:
                vertex_of_same_label += 1

        same_label_percentage.append(vertex_of_same_label/total_vertex)

    return same_label_percentage


################################################################################
#  evaluate the improvement from the incremental learning implementations
################################################################################
def evaluate_improvement(before, after, method):
    if method == 'average':
        ave_before = sum(before)/len(before)
        ave_after = sum(after)/len(after)
        print(ave_before)
        print(ave_after)

    return ave_before, ave_after
