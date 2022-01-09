from __future__ import annotations

import abc
import dataclasses
import functools
import itertools
import sys
from decimal import Decimal
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    Iterator,
    Sequence,
    Tuple,
    TypeVar,
    overload,
)

if TYPE_CHECKING and sys.version_info >= (3, 8):
    from typing import Hashable, Protocol

    from _typeshed import SupportsDunderLT

    class Face(Hashable, SupportsDunderLT, Protocol):
        pass

    class SupportsAddF(Face, Protocol):
        def __add__(self: F, other: F) -> F:
            ...

    class SupportsSubF(Face, Protocol):
        def __sub__(self: F, other: F) -> F:
            ...

    class SupportsMulF(Face, Protocol):
        def __mul__(self: F, other: F) -> F:
            ...

    class SupportsTrueDivF(Face, Protocol):
        def __truediv__(self: F, other: F) -> F:
            ...

    class SupportsFloorDivF(Face, Protocol):
        def __floordiv__(self: F, other: F) -> F:
            ...

    class SupportsModF(Face, Protocol):
        def __mod__(self: F, other: F) -> F:
            ...

    class SupportsFaceDivF(SupportsAddF, SupportsFloorDivF, SupportsModF, Protocol):
        pass

else:
    Face = Any
    SupportsSubF = Any
    SupportsMulF = Any
    SupportsTrueDivF = Any
    SupportsFloorDivF = Any
    SupportsModF = Any
    SupportsFaceDivF = Any


Weight = Decimal
F = TypeVar("F", bound=Face)
F_co = TypeVar("F_co", bound=Face, covariant=True)
G = TypeVar("G", bound=Face)
H = TypeVar("H", bound=Face)


@dataclasses.dataclass
class Faces(Generic[F]):
    weights: Dict[F, Weight]

    @classmethod
    def from_pairs(cls, pairs: Iterable[Tuple[F, Weight]]) -> Faces:
        weights: Dict[F, Weight] = {}
        for f, w in pairs:
            weights.setdefault(f, Decimal())
            weights[f] += w
        return cls(weights)

    @classmethod
    def from_constant(cls, value: F) -> Faces:
        return cls.from_pairs([(value, Decimal(1))])

    def named(self, name: str) -> Die[F]:
        return Named(name, self)

    def __getitem__(self, key: F) -> Weight:
        return self.weights[key]

    def __iter__(self) -> Iterator[Tuple[F, Weight]]:
        return (i for i in self.weights.items())

    def __str__(self) -> str:
        return str({f: float((w * 100).quantize(Decimal("1.000"))) for f, w in self})

    @overload
    def map(self, func: Callable[[F], F]) -> Faces[F]:
        ...

    @overload
    def map(self, func: Callable[[F], G]) -> Faces[G]:
        ...

    def map(self, func):
        return self.from_pairs((func(f), w) for f, w in self)

    @overload
    def combine(self, func: Callable[[F, F], F], other: Faces[F]) -> Faces[F]:
        ...

    @overload
    def combine(self, func: Callable[[F, F], G], other: Faces[F]) -> Faces[G]:
        ...

    @overload
    def combine(self, func: Callable[[F, G], F], other: Faces[G]) -> Faces[F]:
        ...

    @overload
    def combine(self, func: Callable[[F, G], G], other: Faces[G]) -> Faces[G]:
        ...

    @overload
    def combine(self, func: Callable[[F, G], H], other: Faces[G]) -> Faces[H]:
        ...

    def combine(self, func, other):
        return self.from_pairs(
            (func(a_f, b_f), a_w * b_w)
            for (a_f, a_w), (b_f, b_w) in itertools.product(self, other)
        )


def dicefunction(func: Callable[..., Faces[F]]) -> Callable[..., Die[F]]:
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Die[F]:
        return FunctionCall(func, args, kwargs)

    return wrapper


def dicemethod(func: Callable[..., Faces[F]]) -> Callable[..., Die[F]]:
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs) -> Die[F]:
        return FunctionCall(func, (self,) + args, kwargs)

    return wrapper


@dataclasses.dataclass
class Slice(Generic[F]):

    die: Die[F]

    @overload
    def __getitem__(self, key: int) -> Die[F]:
        pass

    @overload
    def __getitem__(self, key: slice) -> Die[Tuple[F, ...]]:
        pass

    def __getitem__(self, key):
        if isinstance(key, int):
            return (
                self.die.slice_values(slice(key, key + 1))
                .faces.map(lambda v: v[0])
                .named(f"{self.die}[{key}]")
            )
        else:
            key_str = (
                ("" if key.start is None else str(key.start))
                + ":"
                + ("" if key.stop is None else str(key.stop))
            )
            if key.step is not None:
                key_str += f":{key.step}"
            return self.die.slice_values(key).named(f"{self.die}[{key_str}]")


class Die(Generic[F], metaclass=abc.ABCMeta):
    @functools.cached_property
    def faces(self) -> Faces:
        return self.get_faces()

    @abc.abstractmethod
    def get_faces(self) -> Faces:
        pass

    @functools.cached_property
    def contained(self) -> Iterable[Die[F]]:
        return list(self.get_contained())

    def get_contained(self) -> Iterable[Die[F]]:
        return []

    @overload
    def combine(self, func: Callable[[F, F], F], other: Die[F]) -> Die[F]:
        ...

    @overload
    def combine(self, func: Callable[[F, F], G], other: Die[F]) -> Die[G]:
        ...

    @overload
    def combine(self, func: Callable[[F, G], F], other: Die[G]) -> Die[F]:
        ...

    @overload
    def combine(self, func: Callable[[F, G], G], other: Die[G]) -> Die[G]:
        ...

    @overload
    def combine(self, func: Callable[[F, G], H], other: Die[G]) -> Die[H]:
        ...

    def combine(self, func, other):
        return self.faces.combine(func, other.faces).named(
            f"{self}.combine({func.__name__}, {other})"
        )

    @overload
    def map(self, func: Callable[[F], F]) -> Die[F]:
        ...

    @overload
    def map(self, func: Callable[[F], G]) -> Die[G]:
        ...

    def map(self, func):
        return self.faces.map(func).named(f"{self}.map({func.__name__})")

    def named(self, name: str) -> Die[F]:
        return Named(name, self.faces)

    def _apply_operator(self, operator, other):
        if isinstance(other, Die):
            return operator(self, other)
        return operator(self, Constant(other))

    def __lshift__(self, other) -> bool:
        # "can merge with"
        return False

    # math methods
    @overload
    def __add__(self, other: Die[F]) -> Die[F]:
        ...

    @overload
    def __add__(self, other: int) -> Die[F]:
        ...

    def __add__(self, other):
        if isinstance(other, Die):
            if self == other:
                return Repeated(2, self)
            if other << self:
                return other + self
            return Bag([self, other])
        return self._apply_operator(add, other)

    def __mul__(self, other: Die[F]) -> Die[F]:
        return self._apply_operator(mul, other)

    def __rmul__(self, other: int) -> Die[F]:
        return Repeated(other, self)

    @overload
    def __sub__(self, other: Die[F]) -> Die[F]:
        ...

    @overload
    def __sub__(self, other: int) -> Die[F]:
        ...

    def __sub__(self, other):
        return self._apply_operator(sub, other)

    @overload
    def __truediv__(self, other: Die[F]) -> Die[F]:
        ...

    @overload
    def __truediv__(self, other: int) -> Die[F]:
        ...

    def __truediv__(self, other):
        return self._apply_operator(truediv, other)

    @overload
    def __floordiv__(self, other: Die[F]) -> Die[F]:
        ...

    @overload
    def __floordiv__(self, other: int) -> Die[F]:
        ...

    def __floordiv__(self, other):
        return self._apply_operator(facediv, other)

    @dicemethod
    def slice_values(
        self, key: slice = slice(0, None), reverse: bool = True
    ) -> Faces[Tuple[F, ...]]:
        acc = Faces.from_constant(tuple())
        for d in self.contained:
            acc = acc.combine(
                lambda a, b: tuple(sorted(a + (b,), reverse=reverse))[: key.stop],
                d.faces,
            )
        return acc.map(lambda f: f[key.start : key.stop : key.step])

    @dicemethod
    def slice(self, key: slice = slice(0, None), reverse: bool = True) -> Faces[F]:
        return self.slice_values(key, reverse).sum().faces

    @dicemethod
    def highest_values(self, n: int = 1) -> Faces[Tuple[F, ...]]:
        return self.slice_values(slice(0, n), reverse=True).faces

    @dicemethod
    def highest(self, n: int = 1) -> Faces[F]:
        return self.highest_values(n).sum().faces

    @dicemethod
    def lowest_values(self, n: int = 1) -> Faces[Tuple[F, ...]]:
        return self.slice_values(slice(0, n), reverse=False).faces

    @dicemethod
    def lowest(self, n: int = 1) -> Faces[F]:
        return self.lowest_values(n).sum().faces

    @dicemethod
    def sum(self) -> Faces[F]:
        return self.faces.map(lambda v: functools.reduce(lambda a, b: a + b, v))

    @property
    def values(self) -> Slice[F]:
        return Slice(self)


@dataclasses.dataclass
class Named(Die[F]):

    name: str
    _faces: Faces[F]

    def get_faces(self) -> Faces[F]:
        return self._faces

    def __str__(self):
        return self.name


@dataclasses.dataclass
class Constant(Die[F]):

    value: F

    def get_faces(self) -> Faces[F]:
        return Faces({self.value: Decimal(1)})

    def __str__(self):
        return str(self.value)


@dataclasses.dataclass(order=True)
class DX(Die[int]):

    size: int

    def get_faces(self) -> Faces[int]:
        return Faces({v: Decimal(1) / self.size for v in range(1, self.size + 1)})

    def __str__(self):
        return f"d{self.size}"


@dataclasses.dataclass
class Seq(Die[F]):

    items: Sequence[F]

    def get_faces(self) -> Faces[F]:
        return Faces({v: Decimal(1) / len(self.items) for v in self.items})

    def __str__(self):
        if isinstance(self.items, range):
            return f"d[{self.items.start}..{self.items.stop-1}]"
        return f"d{list(self.items)}"


@dataclasses.dataclass
class Operator(Generic[F]):
    symbol: str
    precedence: int = 0

    def __call__(
        self, operator: Callable[[F_co, F_co], F_co]
    ) -> Callable[[Die[F_co], Die[F_co]], Die[F_co]]:
        return lambda *args, **kwargs: OperatorCall(
            self.symbol, self.precedence, operator, *args, **kwargs
        )


@dataclasses.dataclass
class OperatorCall(Die[F_co]):

    symbol: str
    precedence: int
    operator: Callable[[F_co, F_co], F_co]
    left: Die[F_co]
    right: Die[F_co]

    def __str__(self):
        def _wrap(die: Die[F_co]) -> str:
            if self.precedence > getattr(die, "precedence", 999):
                return f"({die})"
            return str(die)

        return f"{_wrap(self.left)} {self.symbol} {_wrap(self.right)}"

    def get_faces(self) -> Faces[F_co]:
        return self.left.faces.combine(self.operator, self.right.faces)


@Operator(symbol="+", precedence=10)
def add(left: SupportsAddF, right: SupportsAddF) -> SupportsAddF:
    return left + right


@Operator(symbol="-", precedence=10)
def sub(left: SupportsSubF, right: SupportsSubF) -> SupportsSubF:
    return left - right


@Operator(symbol="*", precedence=11)
def mul(left: SupportsMulF, right: SupportsMulF) -> SupportsMulF:
    return left * right


@Operator(symbol="/", precedence=11)
def truediv(left: SupportsTrueDivF, right: SupportsTrueDivF) -> SupportsTrueDivF:
    return left / right


@Operator(symbol="//", precedence=11)
def facediv(left: SupportsFaceDivF, right: SupportsFaceDivF) -> SupportsFaceDivF:
    return left // right + (1 if left % right else 0)


@dataclasses.dataclass
class Repeated(Die[F]):

    number: int
    die: Die[F]

    def __post_init__(self) -> None:
        if self.number < 1:
            raise ValueError(f"number must be >= 1, got {self.number}")

    def __str__(self) -> str:
        if isinstance(self.die, DX):
            return f"{self.number}{self.die}"
        if isinstance(self.die, Operator):
            return f"{self.number} x ({self.die})"
        return f"{self.number} x {self.die}"

    def get_faces(self) -> Faces[F]:
        ret = self.die.faces
        for _ in range(self.number - 1):
            ret = ret.combine(lambda a, b: a + b, self.die.faces)
        return ret

    def get_contained(self) -> Iterable[Die[F]]:
        return self.number * [self.die]

    def __lshift__(self, other: Die[F]) -> bool:
        return self.die == other

    def __add__(self, other):
        if self << other:
            return dataclasses.replace(self, number=self.number + 1)
        return super().__add__(other)


@dataclasses.dataclass
class Bag(Die[F]):

    dice: Sequence[Die[F]]

    def get_faces(self) -> Faces[F]:
        ret = Faces.from_constant(0)
        for die in self.dice:
            ret = ret.combine(lambda a, b: a + b, die.faces)
        return ret

    def get_contained(self) -> Iterable[Die[F]]:
        return self.dice

    def __str__(self) -> str:
        return " + ".join(str(d) for d in self.dice)

    def __lshift__(self, other: Die[F]) -> bool:
        return True  # can contain anything

    def __add__(self, other):
        dice = list(self.dice)
        for die in dice:
            if not (other == die or other << die):
                continue
            dice.remove(die)
            return dataclasses.replace(
                self,
                dice=dice + [die + other],
            )
        return dataclasses.replace(self, dice=self.dice + [other])


@dataclasses.dataclass
class FunctionCall(Die[F]):

    func: Callable[..., Faces[F]]
    args: Sequence[Die[F]]
    kwargs: Dict[Any, Any]

    def __str__(self):
        parts = [str(arg) for arg in self.args] + [
            f"{k}={v!r}" for k, v in self.kwargs.items()
        ]
        return f"{self.func.__name__}({', '.join(parts)})"

    def get_faces(self) -> Faces:
        return self.func(*self.args, **self.kwargs)


@dicefunction
def explode(die: Die[int], *, n: int = 2) -> Faces[int]:
    if n < 1:
        raise ValueError("must explode with n >= 1, got {n}")

    max_value = max(f for f, _ in die.faces)

    faces = die.faces
    for _ in range(n):
        # If a % max_value == 0, we must have only rolled max values so far.
        faces = faces.combine(lambda a, b: a + (0 if a % max_value else b), die.faces)
    return faces
