"""Rocks fall, everyone dice"""

import abc
import collections.abc
import dataclasses
from decimal import Decimal
import functools
import inspect
import itertools
import operator
import pprint
import re
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    Iterable,
    Optional,
    List,
    Tuple,
    TypeVar,
    Union,
)


Face = TypeVar("Face", bound=Hashable)
A = TypeVar("A", bound=Hashable)
B = TypeVar("B", bound=Hashable)
Coercible = Union[Face, List[Face], "Die"]
WeightInput = Union[int, float, Decimal]

Mapper = Callable[[A], Face]
Reducer = Callable[[A, B], Face]


def print_dice(*args):
    strs = []
    for arg in args:
        if isinstance(arg, AbstractDie):
            strs.append(arg.debug_string)
        else:
            strs.append(str(arg))
    print(*strs)


def facediv(x: int, y: int) -> int:
    """Divide an integer face, rounding up.

    e.g. facediv(5, 2) == 3, the same way you might emulate a d3 with a
    physical d6 (1-2 = 1, 3-4 = 2, 5-6 = 3)
    """
    return x // y + min(x % y, 1)


class AbstractFormat(metaclass=abc.ABCMeta):
    """Abstract base class for die formats."""

    @abc.abstractmethod
    def format(self, die: "Die") -> str:
        """Format the given die."""


class DefaultFormat(AbstractFormat):
    """Default format for dice, based on face values."""

    @classmethod
    def format(cls, die: "Die") -> str:
        faces_list = list(die.faces)
        if len(faces_list) == 1:
            return str(faces_list[0])
        if all(isinstance(face, int) for face in faces_list):
            start = int(min(faces_list))
            stop = int(max(faces_list))
            if sorted(faces_list) == list(range(start, stop + 1)):
                return f"Die({start}..{stop})"
        return f'Die({", ".join(str(f) for f in faces_list)})'


class DXFormat(AbstractFormat):
    """Format for N-sided dice (d4, d6, etc.)."""

    @classmethod
    def format(cls, die: "Die") -> str:
        return f"d{max(die.faces)}"


@dataclasses.dataclass
class ListFormat(AbstractFormat):
    """Format for explicit lists of faces."""

    items: List

    def format(self, die: "Die") -> str:
        start = min(self.items)
        stop = max(self.items)
        if sorted(self.items) == list(range(start, stop + 1)):
            return f"d[{start}..{stop}]"
        return f"d{sorted(self.items)}"


@dataclasses.dataclass
class NamedFormat(AbstractFormat):
    """Format for dice with arbitrary names."""

    name: str

    def format(self, die: "Die") -> str:
        return self.name


@dataclasses.dataclass
class OperatorFormat(AbstractFormat):
    """Format for dice based on operators and their combinations."""

    a: "Die"
    op_func: Callable
    b: "Die"

    OPERATOR_STRING = {
        operator.add: "+",
        operator.sub: "-",
        operator.mul: "*",
        facediv: "//",
        operator.truediv: "/",
    }

    OPERATOR_PRECEDENCE = [
        ["**"],
        ["*", "@", "/", "//", "%"],
        ["+", "-"],
        ["<<", ">>"],
        ["&"],
        ["^"],
        ["|"],
        ["<", "<=", ">", ">=", "!=", "=="],
    ]

    def __post_init__(self):
        if self.op_func not in self.OPERATOR_STRING:
            raise KeyError(f"unhandled operator: {self.op_func}")

    @property
    def op_string(self) -> str:
        """Return the operator's string representation."""
        return self.OPERATOR_STRING[self.op_func]

    @property
    def precedence(self) -> int:
        """Return the operator's precedence.

        Unknown operators are assumed to be the lowest possible, which will
        result in their having parens when combined with other operators.
        """
        i = 999
        for ops in self.OPERATOR_PRECEDENCE:
            if self.op_string in ops:
                return i
            i -= 1
        return i

    def format(self, die: "Die") -> str:
        def _wrap(other: "Die") -> str:
            if (
                isinstance(other.formatter, self.__class__)
                and other.formatter.precedence < self.precedence
            ):
                return f"({other})"
            return str(other)

        return f"{_wrap(self.a)} {self.op_string} {_wrap(self.b)}"


@dataclasses.dataclass
class FunctionFormat(AbstractFormat):

    name: str
    args: List[Any] = dataclasses.field(default_factory=list)
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)
    instance: Optional["AbstractDie"] = None

    def format(self, die: "Die") -> str:
        func_args = ", ".join(
            [str(arg) for arg in self.args]
            + [f"{k}={v!r}" for k, v in self.kwargs.items()]
        )
        if self.instance is not None:
            return f"({self.instance}).{self.name}({func_args})"
        return f"{self.name}({func_args})"

    @classmethod
    def from_caller(cls, func: Callable) -> "FunctionFormat":
        arginfo = inspect.getargvalues(inspect.stack()[1].frame)
        sig = inspect.signature(func)
        bound = sig.bind(
            **{
                name: arginfo.locals[name]
                for name, param in sig.parameters.items()
                if param.default != arginfo.locals[name]
            }
        )
        return cls(
            name=func.__name__,
            args=list(bound.args),
            kwargs=bound.kwargs,
            instance=getattr(func, "__self__", None),
        )


class Faces(Dict[Face, Decimal]):
    def map(self, func: Mapper) -> "Die":
        return Die.from_pairs((func(f), w) for f, w in self.items())

    def reduce(self, func: Reducer) -> "Die":
        return self.map(functools.partial(functools.reduce, func))


class AbstractDie(metaclass=abc.ABCMeta):

    formatter: AbstractFormat

    def named(self, name: str) -> "AbstractDie":
        return dataclasses.replace(self, formatter=NamedFormat(name))

    def formatted(self, formatter: AbstractFormat) -> "AbstractDie":
        return dataclasses.replace(self, formatter=formatter)

    @abc.abstractmethod
    def __str__(self) -> str:
        pass

    @abc.abstractproperty
    def debug_string(self) -> str:
        pass

    @abc.abstractmethod
    def get(self, face: Face) -> Decimal:
        pass

    @abc.abstractmethod
    def apply(self, reducer: Reducer, other: "AbstractDie") -> "AbstractDie":
        pass

    def __add__(self, other: Coercible) -> "AbstractDie":
        if isinstance(self, MultiDie):
            return MultiDie(self.dice + [Die.coerce(other)])
        elif isinstance(self, Die):
            return MultiDie([self, Die.coerce(other)])
        raise TypeError(f"unknown die type: {self!r}")

    def __radd__(self, other: Coercible) -> "AbstractDie":
        return Die.coerce(other) + self

    def __sub__(self, other: Coercible) -> "AbstractDie":
        return self.apply(operator.sub, Die.coerce(other))

    def __rsub__(self, other: Coercible) -> "AbstractDie":
        return Die.coerce(other) - self

    def __mul__(self, other: Coercible) -> "AbstractDie":
        return self.apply(operator.mul, Die.coerce(other))

    def __rmul__(self, other: Coercible) -> "AbstractDie":
        return Die.coerce(other) * self

    def __truediv__(self, other: Coercible) -> "AbstractDie":
        return self.apply(operator.truediv, Die.coerce(other))

    def __rtruediv__(self, other: Coercible) -> "AbstractDie":
        return Die.coerce(other) / self

    def __floordiv__(self, other: Coercible) -> "AbstractDie":
        return self.apply(facediv, Die.coerce(other))

    def __rfloordiv__(self, other: Coercible) -> "AbstractDie":
        return Die.coerce(other) // self

    def __mod__(self, other: Coercible) -> "AbstractDie":
        return self.apply(operator.mod, Die.coerce(other))

    def __rmod__(self, other: Coercible) -> "AbstractDie":
        return Die.coerce(other) % self


@dataclasses.dataclass(frozen=True)
class Die(AbstractDie):
    """A Die is a group of Faces and Weights.

    A Face can be any hashable object; a Weight is a Decimal.
    Identical faces are combined into a single entry with their combined
    weights.
    The sum of weights should always be equal to 1.

    Operators on pairs of dice are applied to their faces elementwise to
    produce a new die, with weights calculated from the input pair of faces'
    weights.

    e.g. as part of evaluating (d6 + d4), these face-weight pairs are added:

    * (1, 0.1666...), (1, 0.25) = (2, 0.041666...)
    * (1, 0.1666...), (2, 0.25) = (3, 0.041666...)
    * [etc.]
    * (6, 0.1666...), (4, 0.25) = (10, 0.041666...)

    The resulting die will have faces valued 2 through 10 with values 5, 6, and
    7 most likely to occur.

    You can construct dice with arbitrary weights, but for most uses, see
    Die.from_iterable(), Die.from_pairs(), and Die.from_value().

    For convenience, operators also promote simple values to dice when
    necessary, e.g. these are identical:

        * d6 + 1
        * d6 + Die.from_value(1)

    Because faces can be arbitrary hashable objects, you can use them to
    represent procedures with historical state or other complex calculations.
    See the tests for examples.
    """

    faces: Dict[Face, Decimal]
    formatter: AbstractFormat = dataclasses.field(default_factory=DefaultFormat)

    def __iter__(self):
        return (i for i in self.faces.items())

    def __str__(self):
        return self.name_string

    def get(self, face: Face) -> Decimal:
        return self.faces[face]

    @property
    def debug_string(self) -> str:
        ret = self.name_string
        faces_str = self.faces_string
        if "\n" in faces_str:
            ret += f":\n{faces_str}"
        else:
            ret += f": {faces_str}"
        return ret

    def __rmatmul__(self, other: int) -> "MultiDie":
        return MultiDie(other * [self])

    @property
    def name_string(self) -> str:
        return self.formatter.format(self)

    @property
    def faces_string(self) -> str:
        return pprint.pformat(
            {f: float((w * 100).quantize(Decimal("1.000"))) for f, w in self}
        )

    @classmethod
    def from_pairs(cls, pairs: Iterable[Tuple[Face, WeightInput]]) -> "Die":
        faces: Dict[Face, List[WeightInput]] = {}
        for f, w in pairs:
            faces.setdefault(f, []).append(w)
        return cls({f: Decimal(sum(w)) for f, w in faces.items()})

    @classmethod
    def from_iterable(cls, iterable: Iterable[Face]) -> "Die":
        faces_list = list(iterable)
        return cls.from_pairs(
            (face, Decimal(1) / len(faces_list)) for face in faces_list
        )

    @classmethod
    def from_value(cls, face: Face) -> "Die":
        return cls({face: Decimal(1)})

    @classmethod
    def coerce(cls, other: Coercible) -> "AbstractDie":
        if isinstance(other, AbstractDie):
            return other
        if isinstance(other, collections.abc.Iterable):
            return cls.from_iterable(other)
        return cls.from_value(other)

    def map_faces(self, func: Mapper) -> "Die":
        return self.from_pairs((func(f), w) for f, w in self)

    def reduce_faces(self, func: Reducer) -> "Die":
        return self.map_faces(functools.partial(functools.reduce, func))

    def apply(self, func: Reducer, other: Coercible) -> "AbstractDie":
        other = Die.coerce(other)
        ret = lift_reducer(func)(self, other)
        try:
            return ret.formatted(OperatorFormat(self, func, other))
        except KeyError:
            pass
        return ret.formatted(FunctionFormat(func.__name__, [self, other]))


@dataclasses.dataclass(frozen=True)
class MultiDie(AbstractDie):
    """A MultiDie is a set of dice whose combinations haven't been evaluated.

    By default, a MultiDie acts like a Die whose faces are generated by summing
    the MultiDie's dice. More complex operations, including getting all the
    individual values rolled, are available by calling reduce() or
    accumulate().
    """

    dice: List[AbstractDie]
    formatter: AbstractFormat = dataclasses.field(default_factory=DefaultFormat)

    def __str__(self) -> str:
        return self.dice_string

    def __iter__(self):
        return (i for i in self.sum)

    def get(self, face: Face) -> Decimal:
        return self.faces[face]

    @property
    def dice_string(self) -> str:
        parts = []
        for die_name, count in collections.Counter(str(d) for d in self.dice).items():
            if count == 1:
                parts.append(die_name)
            elif die_name.startswith("d"):
                # TODO: use better detection
                parts.append(f"{count}{die_name}")
            else:
                parts.append(f"{count}x({die_name})")
        return " + ".join(parts)

    @property
    def debug_string(self) -> str:
        if "\n" in self.sum.faces_string:
            sep = "\n"
        else:
            sep = " "
        return f"{self.dice_string}:{sep}{self.sum.faces_string}"

    @property
    def faces(self) -> Dict[Face, Decimal]:
        return self.sum.faces

    @functools.cached_property
    def sum(self) -> Die:
        return self.reduce(operator.add)

    def apply(self, reducer: Reducer, other: AbstractDie) -> Die:
        raise NotImplementedError()

    def reduce(self, reducer: Reducer, /, initial: Optional[Die] = None) -> Die:
        reduce_args = [lift_reducer(reducer), self.dice]
        if initial is not None:
            reduce_args.append(initial)
        return functools.reduce(*reduce_args).formatted(
            FunctionFormat(reducer.__name__, self.dice)
        )

    def slice(
        self,
        start: int = 0,
        stop: Optional[int] = None,
        step: int = 1,
        reverse: bool = False,
    ) -> Die:
        return self.reduce(
            lambda a, b: tuple(sorted(a + (b,), reverse=reverse)[:stop]),
            initial=Die.from_value(tuple()),
        ).map_faces(lambda v: v[start:stop:step])

    def highest_values(self, n: int = 1) -> Die:
        return self.slice(0, n, reverse=True)

    def highest(self, n: int = 1) -> Die:
        return (
            self.highest_values(n)
            .map_faces(sum)
            .formatted(FunctionFormat.from_caller(self.highest))
        )

    def lowest_values(self, n: int = 1) -> Die:
        return self.slice(0, n, reverse=False)

    def lowest(self, n: int = 1) -> Die:
        return self.lowest_values(n).map_faces(sum)


def lift_reducer(func: Reducer) -> Callable[[Die, Die], AbstractDie]:
    """Lift a reducer function from faces to dice.

    A reducer function combines two faces and returns a new face.
    This function wraps a reducer so that it applies to each possible
    combination of faces from two dice.
    """

    @functools.wraps(func)
    def wrapper(a: Die, b: Die) -> AbstractDie:
        return Die.from_pairs(
            (func(a_f, b_f), a_w * b_w)
            for (a_f, a_w), (b_f, b_w) in itertools.product(a, b)
        )

    return wrapper


def explode(die: AbstractDie, *, n: int = 2) -> AbstractDie:
    """Return an exploding version of the given die.

    An exploding die is one that, whenever it rolls the maximum value, re-rolls
    and adds the new value as well, continuing to roll and add as long as the
    maximum is rolled.

    By default, only two explosions will be calculated.
    """
    fmt = FunctionFormat.from_caller(explode)

    if n < 1:
        raise ValueError(f"must explode with n at least 1, got {n}")

    def reducer(a: int, b: int) -> int:
        if a % max(die.faces) == 0:
            return a + b
        return a

    acc = die
    # TODO: replace with MultiDie method
    for _ in range(n):
        acc = acc.apply(reducer, die)

    return acc.formatted(fmt)


def parse(spec: str) -> AbstractDie:
    """Parse dice strings.

    Dice are of the form "<N>d<X>" where N is optional, e.g. "3d6", "d8".

    Multiple kinds of dice can be specified by using "+", e.g. "2d6+d4". Other
    operators are not supported.
    """
    dice = []
    for dice_spec in re.split(r"\s*\+\s*", spec):
        n, x = dice_spec.split("d")
        try:
            dice.extend(int(n or "1") * [d(int(x))])
        except ValueError as exc:
            raise ValueError(f"while parsing {dice_spec}: {exc}") from exc
    if len(dice) == 1:
        return dice[0]
    return MultiDie(dice)


class D:
    def __call__(self, arg: Union[Iterable, int, str]) -> AbstractDie:
        if isinstance(arg, int):
            return Die.from_iterable(range(1, arg + 1)).formatted(DXFormat())
        if isinstance(arg, str):
            return parse(arg)
        if isinstance(arg, collections.abc.Iterable):
            faces = list(arg)
            return Die.from_iterable(faces).formatted(ListFormat(faces))
        raise ValueError(f"unhandled die type: {arg}")

    def __getattr__(self, attr):
        if attr.startswith("_"):
            return parse(attr[1:])
        return parse(attr)


def d(arg: Union[Iterable, int, str]) -> AbstractDie:  # pylint: disable=invalid-name
    """Smart dice constructor.

    Options:
        * An Iterable will be passed to Die.from_iterable.
        * A string will be passed to parse() to generate a Die or MultiDie.
        * An int 'n' will be used to generate a list of numbers (1..n) and
          passed to Die.from_iterable.
    """
    if isinstance(arg, int):
        return Die.from_iterable(range(1, arg + 1)).formatted(DXFormat())
    if isinstance(arg, str):
        return parse(arg)
    if isinstance(arg, collections.abc.Iterable):
        faces = list(arg)
        return Die.from_iterable(faces).formatted(ListFormat(faces))
    raise ValueError(f"unhandled die type: {arg}")


_d = D()


if __name__ == "__main__":
    d4 = d(4)
    d6 = d(6)
    d8 = d(8)
    dF = d([-1, 0, +1]).named("dF")
    print_dice(d6)
    print_dice(d6 / 2)
    print_dice(d6 // 2)
    print_dice(d6 + d6)
    print_dice(2 @ d6 + 3 @ d4 + 1 @ d8)
    print_dice(d("2d6 + 3d4 + d8"))
    print_dice(explode(d6, n=3))
    print_dice(explode(d6) + d4)
    print_dice(3 * (d6 + 1))
    print_dice(1 + d6)
    #    for i in d6.faces:
    #        for j in (d4 + 1).faces:
    #            print_dice(f'facediv({i}, {j}) = {facediv(i, j)}')
    print_dice(d6 // (d4 + 1))
    print_dice(1 - d6)
    print_dice(d8)
    print_dice(3 @ dF)
