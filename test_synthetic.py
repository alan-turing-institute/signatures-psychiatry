import os
import pytest
import re

@pytest.mark.parametrize("cohort, expected_values",
                         [("239673", [0.828283, 0.920596, 0.785203, 0.828336, 0.916247, 0.76653]),
                          ("425769", [0.810101, 0.905707, 0.77327,  0.809524, 0.907198, 0.744932]),
                          ("772192", [0.836364, 0.903226, 0.74463,  0.836493, 0.899675, 0.722961])])
def test_pairwise_group_classification_synth(cohort, expected_values, capfd):
    """
    Check that classification of the three synthetic cohorts with the default seed is consistent.
    Values in expected_values are:
        accuracy for healthy-bipolar, healthy-borderline and bipolar-borderline
        area under ROC curve for healthy-bipolar, healthy-borderline and bipolar-borderline
    """

    # Run pairwise_group_classification on each synthetic cohort and capture stdout (which contains results)
    os.system("python pairwise_group_classification.py --synth=" + cohort)
    capture = capfd.readouterr().out

    # Extract the values from stdout - there should only be six non-integer numbers in the output
    acc_auc = re.findall(r'\d\.\d+', capture)
    assert(len(acc_auc) == 6)

    for expected, actual in zip(expected_values, acc_auc):
        assert(expected == float(actual))


def test_pairwise_group_classification_synth_defaults(capfd):
    """
    Check that the default settings used in pairwise_group_classification.py are as expected.
    We'll only check options that use the synthetic datasets.
    """

    # Run pairwise_group_classification on the default synthetic cohort and capture stdout (which contains settings)
    os.system("python pairwise_group_classification.py --synth")
    capture = capfd.readouterr().out

    # Check random seed
    seed = re.findall(r'Random seed has been set to (\d+)', capture)
    assert(len(seed) == 1)
    assert(int(seed[0]) == 83042)

    # Check default cohort
    cohort = re.findall(r'Preparing to load synthetic signatures from cohort (\d+)', capture)
    assert(len(cohort) == 1)
    assert(int(cohort[0]) == 772192)
