import unittest

from mitoolspro.plotting.plots.validator_helper import apply_validators
from mitoolspro.plotting.plots.validation.models import (
    ColorParam,
    ColorSequenceParam,
    ColorSequencesParam,
    NumericParam,
    NumericSequenceParam,
    NumericSequencesParam,
)


class TestApplyValidators(unittest.TestCase):
    def test_first_validator_success(self):
        value, errors = apply_validators(
            [[1, 2], [3, 4]],
            [NumericSequencesParam, NumericSequenceParam, NumericParam],
            sizes=2,
            sub_sizes=[2, 2],
        )
        self.assertEqual(value, [[1, 2], [3, 4]])
        self.assertEqual(errors, [])

    def test_second_validator_success(self):
        value, errors = apply_validators(
            [1, 2, 3],
            [NumericSequencesParam, NumericSequenceParam, NumericParam],
            sizes=3,
            sub_sizes=None,
        )
        self.assertEqual(value, [1, 2, 3])
        self.assertEqual(len(errors), 1)

    def test_last_validator_success(self):
        value, errors = apply_validators(
            "red",
            [ColorSequencesParam, ColorSequenceParam, ColorParam],
            sizes=1,
        )
        self.assertEqual(value, "red")
        self.assertEqual(len(errors), 2)


if __name__ == "__main__":
    unittest.main()
