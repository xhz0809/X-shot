from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score


def accu_metric_fewrel(prediction_prob_list,number_relations):

    num_instance_group = int(len(prediction_prob_list) / number_relations)
    c = 0

    for i in range(num_instance_group):
        start, end = i*number_relations , (i+1)*number_relations
        sublist = prediction_prob_list[start:end]
        assert len(sublist) == number_relations
        max_index = sublist.index(max(sublist))
        if max_index == 0:
            c += 1

    accu = c / num_instance_group

    metrics = {}
    metrics['accuracy'] = accu

    return metrics



def accu_metric_Maven(gold_label_list, positive_prob_list, threshold):

    number_types = 69 # and the "None" label
    num_instance_group = int(len(positive_prob_list) / number_types) # how many dev/test instances 
    assert len(gold_label_list) == len(positive_prob_list)

    gold_label = []
    predicted_label = []
    for i in range(num_instance_group):
        start, end = i*number_types, (i+1)*number_types

        gold_label_sublist = gold_label_list[start:end]
        gold_label_sublist.append(1) if 1 not in gold_label_sublist else gold_label_sublist.append(0)
        max_index = gold_label_sublist.index(max(gold_label_sublist))
        gold_label.append(max_index)

        positive_prob_sublist = positive_prob_list[start:end]
        if max(gold_label_sublist) >= threshold:
            max_index = positive_prob_sublist.index(max(positive_prob_sublist))
        else:
            max_index = 69
        predicted_label.append(max_index)
    assert len(predicted_label) == num_instance_group
    
    group_split = [0, 23*100, 23*100*2, 23*100*2+24*100]

    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true=gold_label, y_pred=predicted_label)

    metric_list = ["freq_accu", "few_accu", "zero_accu"]
    for i in range(3):
        metric_name = metric_list[i]
        start, end = group_split[i], group_split[i+1]
        predicted_sublist = predicted_label[start:end]
        gold_sublist = gold_label[start:end]
        accu = accuracy_score(y_true=gold_sublist, y_pred=predicted_sublist)
        metrics[metric_name] = accu
    
    print(f"accuracy dict: {metrics}")
    return metrics



def accu_metric_RAMS(gold_label_list, positive_prob_list):

    number_types = 30
    num_instance_group = int(len(positive_prob_list) / number_types) # how many dev/test instances 
    assert len(gold_label_list) == len(positive_prob_list)

    gold_label = []
    predicted_label = []
    for i in range(num_instance_group):
        start, end = i*number_types, (i+1)*number_types

        gold_label_sublist = gold_label_list[start:end]
        max_index = gold_label_sublist.index(max(gold_label_sublist))
        assert max(gold_label_sublist) == 1
        gold_label.append(max_index)

        positive_prob_sublist = positive_prob_list[start:end]
        max_index = positive_prob_sublist.index(max(positive_prob_sublist))
        predicted_label.append(max_index)
    assert len(predicted_label) == len(gold_label) == num_instance_group

    
    group_split = [0, 10*50, 10*50*2, 10*50*3]

    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true=gold_label, y_pred=predicted_label)

    metric_list = ["freq_accu", "few_accu", "zero_accu"]
    for i in range(3):
        metric_name = metric_list[i]
        start, end = group_split[i], group_split[i+1]
        predicted_sublist = predicted_label[start:end]
        gold_sublist = gold_label[start:end]
        accu = accuracy_score(y_true=gold_sublist, y_pred=predicted_sublist)
        metrics[metric_name] = accu
    
    print(f"accuracy dict: {metrics}")
    return metrics





