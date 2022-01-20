# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

import unittest
from decimal import Decimal

from parameterized import parameterized  # type: ignore

from rocks_fall import d
from rocks_fall import dice


_d4 = d(4)
_d6 = d(6)
_2d6 = 2 * _d6
_3d6 = 3 * _d6


class FacesTest(unittest.TestCase):
    def test_from_pairs_handles_duplicates(self):
        pairs = (
            (0, Decimal(0.25)),
            (0, Decimal(0.25)),
            (1, Decimal(0.5)),
        )
        faces = dice.Faces.from_pairs(pairs)
        self.assertEqual(faces, dice.Faces({0: Decimal(0.5), 1: Decimal(0.5)}))

    def test_map_handles_duplicates(self):
        faces = dice.Faces({1: Decimal(0.5), 2: Decimal(0.5)})
        self.assertEqual(faces.map(lambda x: not x), d.constant(False).faces)


class DiceTest(unittest.TestCase):
    def test_slice_beyond_number_of_dice(self):
        self.assertEqual(_d6.highest(2).faces, _d6.highest().faces)

    def test_len(self):
        self.assertEqual(len(_d6), 6)
        self.assertEqual(len(d[1, 1, 2, 3, 5]), 4)
        self.assertEqual(len(_2d6), 11)

    @parameterized.expand(
        [
            ("two dice sum", (_d6 + _d6).faces[8], 13.889),
            ("whole slice", _2d6.values[:].faces[(6, 6)], 2.778),
            ("highest", _2d6.highest().faces[6], 30.556),
            ("lowest", _2d6.lowest().faces[6], 2.778),
            (
                "highest values",
                _3d6.values[:2].faces[(6, 6)],
                7.407,
            ),
            ("lowest values", _3d6.values[-2:].faces[(6, 6)], 0.463),
            (
                "middle slice",
                _3d6.values[1].faces[3],
                24.074,
            ),
            ("facediv", (_d6 // 2).faces[1], 33.333),
            ("comparison", (_d6 >= 5).faces[True], 33.333),
        ]
    )
    def test_weights(self, _, weight, expected):
        self.assertAlmostEqual(float(weight * 100), float(expected), places=3)

    @parameterized.expand(
        [
            ("dX", _d6, "d6"),
            ("d[range]", d[range(5, 11)], "d[5..10]"),
            ("d[...]", d[1, 1, 2, 3, 5], "d[1, 1, 2, 3, 5]"),
            ("named", d[-1, 0, 1].named("dF"), "dF"),
            ("operator", _d6 - 1, "d6 - 1"),
            ("operator + parens", (_d6 - 1) // 3, "(d6 - 1) // 3"),
            ("operator + parens (bag)", (_d4 + _d6) * 3, "(d4 + d6) * 3"),
            ("operator with id value", _d6 + 0, "d6"),
            ("operator with opposite sign", _d6 + (-1), "d6 - 1"),
            ("operator with opposite sign (-)", _d6 - (-1), "d6 + 1"),
            ("comparison operator", _d6 >= 4, "d6 >= 4"),
            ("repetition is self", 1 * _d6, "d6"),
            ("repetition with *", 2 * _d6, "2d6"),
            ("repetition with +, single dice (2)", (d(6) + d(6)), "2d6"),
            ("repetition with +, repeated dice", (2 * _d6 + _d6), "3d6"),
            ("repetition with +, repeated dice (r)", (_d6 + 2 * _d6), "3d6"),
            ("bag, two dice", _d6 + _d4, "d6 + d4"),
            ("bag, more dice", _2d6 + _d4, "2d6 + d4"),
            ("bag, mixed + order", _d6 + _d4 + _d6 + _d4, "2d6 + 2d4"),
            ("bag with id value", _d6 + _d4 + 0, "d6 + d4"),
            ("function", dice.explode(_d6), "explode(d6)"),
            ("function + args", dice.explode(_d6, n=3), "explode(d6, n=3)"),
            ("method", _3d6.highest(), "3d6.highest()"),
            ("method + arg", _3d6.highest(2), "3d6.highest(2)"),
            ("method, self parens", (_d6 + _d4).highest(), "(d6 + d4).highest()"),
            ("getitem, single key", _3d6.values[0], "3d6[0]"),
            ("getitem, slice", _3d6.values[:2], "3d6[:2]"),
            ("getitem, negative slice", _3d6.values[-2:], "3d6[-2:]"),
            ("getitem, middle slice", (4 * _d6).values[1:3], "4d6[1:3]"),
            ("getitem, slice with step", (4 * _d6).values[:4:2], "4d6[:4:2]"),
            ("getitem of operator", (_d6 + 3 * _d4).values[:2], "(d6 + 3d4)[:2]"),
        ]
    )
    def test_str(self, _, die, expected):
        self.assertEqual(str(die), expected)
