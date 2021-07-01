from cropharvest.utils import deterministic_shuffle


def test_deterministic_shuffle():

    input_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    all_outputs = []
    # also, test a seed which is much larger than the list length
    for seed in list(range(10)) + [42]:
        all_outputs.append(deterministic_shuffle(input_list, seed))
        assert len(all_outputs[-1]) == len(input_list)
        assert len(set(all_outputs[-1])) == len(set(input_list))

    for i in range(1, len(all_outputs)):
        assert all_outputs[0] != all_outputs[i]
