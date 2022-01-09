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
import sys
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Generic,
    Hashable,
    Iterable,
    Optional,
    List,
    Mapping,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    overload,
    TYPE_CHECKING,
)

if TYPE_CHECKING and sys.version_info >= (3, 8):
    from _typeshed import SupportsDunderLT
    from typing import Protocol

    class BaseFace(Hashable, SupportsDunderLT, Protocol):
        pass

    F = TypeVar("F", bound=BaseFace, covariant=True)
    F2 = TypeVar("F2", bound=BaseFace, covariant=True)
    F3 = TypeVar("F3", bound=BaseFace, covariant=True)

#    class BaseDie(Protocol[F]):
#        @abc.abstractproperty
#        def faces(self) -> 'Iterator[Tuple[F, Decimal]]':
#            raise NotImplementedError
#
#        def apply(self, func: Callable[[F, F2], F3], other: 'BaseDie[F2]') -> 'BaseDie[F3]':
#            ret = lift_reducer(func)(self, other)
#            # Set operator or function name
#            return ret


else:
    BaseFace = Any
    BaseDie = Any


Face = TypeVar("Face", bound=BaseFace)
A = TypeVar("A", bound=BaseFace)
FA = TypeVar("FA", bound=BaseFace)
B = TypeVar("B", bound=BaseFace)
FB = TypeVar("FB", bound=BaseFace)
FC = TypeVar("FC", bound=BaseFace)
D = TypeVar("D", bound="AbstractDie", covariant=True)
Coercible = Union[Face, Iterable[Face], "Die"]
WeightInput = Union[int, float, Decimal]
FaceWeightPairs = Iterable[Tuple[Face, Decimal]]

Reducer = Callable[[A, B], Face]


def print_dice(*args) -> None:
    strs = []
    for arg in args:
        if isinstance(arg, AbstractDie):
            strs.append(arg.debug_string)
        else:
            strs.append(str(arg))
    print(*strs)


def facediv(x: Face, y: Face) -> Face:
    """Divide a numerical face, rounding up.

    e.g. facediv(5, 2) == 3, the same way you might emulate a d3 with a
    physical d6 (1-2 = 1, 3-4 = 2, 5-6 = 3)
    """
    return operator.floordiv(x, y) + min(operator.mod(x, y), 1)


class AbstractFormat(metaclass=abc.ABCMeta):
    """Abstract base class for die formats."""

    @abc.abstractmethod
    def format(self, die: "Die") -> str:
        """Format the given die."""


class DefaultFormat(AbstractFormat):
    """Default format for dice, based on face values."""

    @classmethod
    def format(cls, die: "Die") -> str:
        faces_list = list(f for f, _ in die.faces)
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

    def format(self, die: "AbstractDie") -> str:
        return self.name


@dataclasses.dataclass
class OperatorFormat(AbstractFormat):
    """Format for dice based on operators and their combinations."""

    a: "AbstractDie"
    op_func: Callable
    b: "AbstractDie"

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

    def __post_init__(self) -> None:
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

    def format(self, die: "AbstractDie") -> str:
        def _wrap(other: "AbstractDie") -> str:
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
    args: Sequence[Any] = dataclasses.field(default_factory=list)
    kwargs: Mapping[str, Any] = dataclasses.field(default_factory=dict)
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


class Faces(List[Tuple[Face, Decimal]]):
    @overload
    def map(self, func: Callable[[Face], Face]) -> "Die[Face]":
        ...

    @overload
    def map(self, func: Callable[[Face], FB]) -> "Die[FB]":
        ...

    def map(self, func):
        return Die.from_pairs((func(f), w) for f, w in self)

    def sum(self) -> 'Die[Face]':
        return self.map(lambda vs: functools.reduce(operator.add, vs))


class AbstractDie(Generic[Face], metaclass=abc.ABCMeta):

    formatter: AbstractFormat

    def named(self: D, name: str) -> D:
        return dataclasses.replace(self, formatter=NamedFormat(name))

    def formatted(self: D, formatter: AbstractFormat) -> D:
        return dataclasses.replace(self, formatter=formatter)

    @abc.abstractmethod
    def __str__(self) -> str:
        pass

    @abc.abstractproperty
    def debug_string(self) -> str:
        pass

    @abc.abstractproperty
    def faces(self) -> Faces:
        pass

    def get(self, key: Face) -> Decimal:
        return dict(self.faces)[key]

    @abc.abstractmethod
    def apply(
        self, func: Callable[[Face, FB], FC], other: "AbstractDie[FB]"
    ) -> "AbstractDie[FC]":
        pass

    def accumulate(
        self, func: Callable[[Face, FB], Face], others: "Iterable[AbstractDie[FB]]"
    ) -> Generator["AbstractDie[Face]", None, None]:
        ret = self
        for other in others:
            ret = ret.apply(func, other)
            yield ret

    @overload
    @classmethod
    def coerce(cls, other: "AbstractDie") -> "AbstractDie":  # type: ignore
        pass

    @overload
    @classmethod
    def coerce(cls, other: Iterable[FB]) -> "Die[FB]":
        pass

    @overload
    @classmethod
    def coerce(cls, other: FB) -> "Die[FB]":
        pass

    @classmethod
    def coerce(cls, other):
        if isinstance(other, AbstractDie):
            return other
        if isinstance(other, collections.abc.Iterable):
            return Die.from_iterable(other)
        return Die.single_value(other)

    def __add__(self, other: Coercible) -> "AbstractDie":
        if isinstance(self, MultiDie):
            return MultiDie(list(self.dice) + [self.coerce(other)])
        elif isinstance(self, Die):
            return MultiDie([self, self.coerce(other)])
        raise TypeError(f"unknown die type: {self!r}")

    def __radd__(self, other: Coercible) -> "AbstractDie":
        return self.coerce(other) + self

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
class Die(AbstractDie[Face]):
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
    Die.from_iterable(), Die.from_pairs(), and Die.single_value().

    For convenience, operators also promote simple values to dice when
    necessary, e.g. these are identical:

        * d6 + 1
        * d6 + Die.single_value(1)

    Because faces can be arbitrary hashable objects, you can use them to
    represent procedures with historical state or other complex calculations.
    See the tests for examples.
    """

    _faces: Faces
    formatter: AbstractFormat = dataclasses.field(default_factory=DefaultFormat)

    def __str__(self):
        return self.name_string

    @property
    def debug_string(self) -> str:
        ret = self.name_string
        faces_str = self.faces_string
        if "\n" in faces_str:
            ret += f":\n{faces_str}"
        else:
            ret += f": {faces_str}"
        return ret

    @property
    def faces(self) -> Faces:
        return self._faces

    def __rmatmul__(self, other: int) -> "MultiDie[Face]":
        return MultiDie(other * [self])

    @property
    def name_string(self) -> str:
        return self.formatter.format(self)

    @property
    def faces_string(self) -> str:
        return pprint.pformat(
            {f: float((w * 100).quantize(Decimal("1.000"))) for f, w in self.faces}
        )

    @classmethod
    def from_pairs(cls, pairs: Iterable[Tuple[Face, WeightInput]]) -> "Die[Face]":
        faces: Dict[Face, List[WeightInput]] = {}
        for f, w in pairs:
            faces.setdefault(f, []).append(w)
        return cls(Faces([(f, Decimal(sum(w))) for f, w in faces.items()]))

    @classmethod
    def from_iterable(cls, iterable: Iterable[Face]) -> "Die[Face]":
        faces_list = list(iterable)
        return cls.from_pairs(
            (face, Decimal(1) / len(faces_list)) for face in faces_list
        )

    @classmethod
    def single_value(cls, face: Face) -> "Die[Face]":
        return cls.from_iterable([face])

    def apply(
        self, func: Callable[[Face, FB], FC], other: "AbstractDie[FB]"
    ) -> "Die[FC]":
        ret = lift_reducer(func)(self, other)
        try:
            return ret.formatted(OperatorFormat(self, func, other))
        except KeyError:
            pass
        return ret.formatted(FunctionFormat(func.__name__, [self, other]))


@dataclasses.dataclass(frozen=True)
class MultiDie(AbstractDie[Face]):
    """A MultiDie is a set of dice whose combinations haven't been evaluated.

    By default, a MultiDie acts like a Die whose faces are generated by summing
    the MultiDie's dice. More complex operations, including getting all the
    individual values rolled, are available by calling reduce() or
    accumulate().
    """

    dice: Sequence[AbstractDie[Face]]
    formatter: AbstractFormat = dataclasses.field(default_factory=DefaultFormat)

    def __str__(self) -> str:
        return self.dice_string

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
    def faces(self) -> Faces:
        return self.sum.faces

    @functools.cached_property
    def sum(self) -> Die:
        return self.reduce(operator.add)

    def apply(self, reducer: Reducer, other: AbstractDie) -> Die:
        raise NotImplementedError()

    @overload
    def reduce(
        self,
        reducer: Callable[[Face, Face], Face],
        /,
        initial: Optional[Die[Face]] = None,
    ) -> Die[Face]:
        pass

    @overload
    def reduce(
        self, reducer: Callable[[FB, Face], FC], /, initial: Optional[Die[FB]] = None
    ) -> Die[FC]:
        pass

    def reduce(self, reducer, /, initial=None):
        if initial is not None:
            ret = functools.reduce(lift_reducer(reducer), self.dice, initial)
        else:
            ret = functools.reduce(lift_reducer(reducer), self.dice)
        return ret.formatted(FunctionFormat(reducer.__name__, self.dice))

    def slice(
        self,
        start: int = 0,
        stop: Optional[int] = None,
        step: int = 1,
        reverse: bool = False,
    ) -> Die[Tuple[Face, ...]]:
        return self.reduce(
            # We need protocols to represent this properly.
            # This line means that all faces need to be sortable. If we could
            # get away from that or relax it, things would be simpler.
            lambda a, b: tuple(sorted(a + (b,), reverse=reverse)[:stop]),
            initial=Die.single_value(tuple()),
        ).faces.map(lambda v: v[start:stop:step])

    def highest_values(self, n: int = 1) -> Die[Tuple[Face, ...]]:
        return self.slice(0, n, reverse=True)

    def highest(self, n: int = 1) -> Die[Face]:
        return (
            self.highest_values(n)
            .faces.map(lambda vs: functools.reduce(lambda a, b: a + b, vs))
            .formatted(FunctionFormat.from_caller(self.highest))
        )

    def lowest_values(self, n: int = 1) -> Die[Tuple[Face, ...]]:
        return self.slice(0, n, reverse=False)

    def lowest(self, n: int = 1) -> Die[Face]:
        return (
            self.lowest_values(n)
            .faces.map(lambda vs: functools.reduce(lambda a, b: a + b, vs))
            .formatted(FunctionFormat.from_caller(self.lowest))
        )


def lift_reducer(
    func: Callable[[FA, FB], FC]
) -> Callable[[AbstractDie[FA], AbstractDie[FB]], Die[FC]]:
    """Lift a reducer function from faces to dice.

    A reducer function combines two faces and returns a new face.
    This function wraps a reducer so that it applies to each possible
    combination of faces from two dice.
    """

    @functools.wraps(func)
    def wrapper(a: AbstractDie[FA], b: AbstractDie[FB]) -> Die[FC]:
        return Die.from_pairs(
            (func(a_f, b_f), a_w * b_w)
            for (a_f, a_w), (b_f, b_w) in itertools.product(a.faces, b.faces)
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
        if a % max(f for f, _ in die.faces) == 0:
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


@overload
def d(arg: Iterable[Face]) -> Die[Face]:
    pass


@overload
def d(arg: int) -> Die[int]:
    pass


def d(arg):  # pylint: disable=invalid-name
    """Smart dice constructor.

    Options:
        * An Iterable will be passed to Die.from_iterable.
        * An int 'n' will be used to generate a list of numbers (1..n) and
          passed to Die.from_iterable.
    """
    if isinstance(arg, int):
        return Die.from_iterable(range(1, arg + 1)).formatted(DXFormat())
    if isinstance(arg, collections.abc.Iterable):
        faces = list(arg)
        return Die.from_iterable(faces).formatted(ListFormat(faces))
    raise ValueError(f"unhandled die type: {arg}")


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
