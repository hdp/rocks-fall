# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

import unittest
from decimal import Decimal

from parameterized import parameterized

from rocks_fall import dice

_d6 = dice.d(6)
_2d6 = 2 @ _d6
_3d6 = 3 @ _d6


class DiceTest(unittest.TestCase):

    def test_from_pairs_handles_duplicates(self):
        pairs = (
            (0, Decimal(0.25)),
            (0, Decimal(0.25)),
            (1, Decimal(0.5)),
        )
        die = dice.Die.from_pairs(pairs)
        self.assertEqual(die.faces, {0: Decimal(0.5), 1: Decimal(0.5)})

    def test_map_handles_duplicates(self):
        die = dice.Die({1: Decimal(0.5), 2: Decimal(0.5)})
        self.assertEqual((die // 2).faces, {1: Decimal(1)})

    @parameterized.expand(
        [
            ("single die", "d6", _d6),
            ("multi die", "3d6", _3d6),
        ]
    )
    def test_parse(self, _, spec, expected):
        self.assertEqual(dice.parse(spec).faces, expected.faces)

    @parameterized.expand(
        [
            ("two dice sum", (_d6 + _d6).get(8), 13.889),
            ("whole slice", _2d6.slice().get((6, 6)), 2.778),
            ("highest", _2d6.highest().get(6), 30.556),
            ("lowest", _2d6.lowest().get(6), 2.778),
            (
                "highest values",
                _3d6.highest_values(2).get((6, 6)),
                7.407,
            ),
            ("lowest values", _3d6.lowest_values(2).get((6, 6)), 0.463),
            (
                "middle slice",
                _3d6.slice(1, 2).map_faces(sum).get(3),
                24.074,
            ),
        ]
    )
    def test_weights(self, _, weight, expected):
        self.assertAlmostEqual(float(weight * 100), float(expected), places=3)

    @parameterized.expand(
        [
            ("dX", _d6, "d6"),
            ("d[range]", dice.d(range(5, 11)), "d[5..10]"),
            ("d[...]", dice.d([1, 1, 2, 3, 5]), "d[1, 1, 2, 3, 5]"),
            ("named", dice.d([-1, 0, 1]).named("dF"), "dF"),
            ("operator", _d6 - 1, "d6 - 1"),
            ("operator + parens", (_d6 - 1) // 3, "(d6 - 1) // 3"),
            ("function", dice.explode(_d6), "explode(d6)"),
            ("function + args", dice.explode(_d6, n=3), "explode(d6, n=3)"),
            # TODO: Maybe make this 'highest(1 of 3d6)'
            ("method", _3d6.highest(), "(3d6).highest()"),
        ]
    )
    def test_format(self, _, die, expected):
        self.assertEqual(str(die), expected)
