import loader


def test_clip_data():
    positive_examples = ['a', 'b', 'c']
    negative_examples = ['d', 'e', 'f', 'g']

    split_positive, split_negative = loader._clip_data(
        positive_examples, negative_examples, 0.5, 7)
    assert len(split_positive) == len(split_negative) == 3

    split_positive, split_negative = loader._clip_data(
        positive_examples, negative_examples, 0.5, 4)
    assert len(split_positive) == len(split_negative) == 2

    split_positive, split_negative = loader._clip_data(
        positive_examples, negative_examples, 0.75, 4)
    assert len(split_positive) == 3
    assert len(split_negative) == 1
