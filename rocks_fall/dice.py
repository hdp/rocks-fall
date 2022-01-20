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

    class SupportsFaceDivF(SupportsAddF, Protocol):
        def __divmod__(self: F, other: F) -> Tuple[F, F]:
            ...

    class SupportsLtF(Face, Protocol):
        def __lt__(self: F, other: F) -> F:
            ...

    class SupportsLeF(Face, Protocol):
        def __le__(self: F, other: F) -> F:
            ...

    class SupportsEqF(Face, Protocol):
        def __eq__(self: F, other: F) -> F:  # type: ignore
            ...

    class SupportsNeF(Face, Protocol):
        def __ne__(self: F, other: F) -> F:  # type: ignore
            ...

    class SupportsGtF(Face, Protocol):
        def __gt__(self: F, other: F) -> F:
            ...

    class SupportsGeF(Face, Protocol):
        def __ge__(self: F, other: F) -> F:
            ...

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


_ADD = 20
_SUB = 20
_MUL = 30
_DIV = 30
_CMP = 10
_NO_PARENS_NEEDED = 99


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
        return MethodCall(self, func, args, kwargs)

    return wrapper


@dataclasses.dataclass
class Slicer(Generic[F]):

    die: Die[F]

    @overload
    def __getitem__(self, key: int) -> Die[F]:
        pass

    @overload
    def __getitem__(self, key: slice) -> Die[Tuple[F, ...]]:
        pass

    def __getitem__(self, key):
        if isinstance(key, int):
            return Item(self.die, key)
        return Slice(self.die, key)


class Die(Generic[F], metaclass=abc.ABCMeta):

    # Subclasses should override this to be automatically wrapped in parens
    # when formatting as needed.
    def get_precedence(self) -> int:
        return _NO_PARENS_NEEDED

    def maybe_wrap(self, other: Die) -> str:
        if self.get_precedence() > other.get_precedence():
            return f"({other})"
        return str(other)

    @abc.abstractmethod
    def __str__(self) -> str:
        pass

    @property
    def faces(self) -> Faces:
        return self.get_faces()

    @abc.abstractmethod
    def get_faces(self) -> Faces:
        pass

    def __len__(self) -> int:
        return len(self.faces.weights)

    @property
    def contained(self) -> Iterable[Die[F]]:
        return list(self.get_contained())

    def get_contained(self) -> Iterable[Die[F]]:
        return [self]

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
        return other in self.contained

    # math methods
    @overload
    def __add__(self, other: Die[F]) -> Die[F]:
        ...

    @overload
    def __add__(self, other: F) -> Die[F]:
        ...

    def __add__(self, other):
        if isinstance(other, Die):
            if self << other:
                # The base class implementation of << is equality
                return Repeated(2, self)
            if other << self:
                # Another class might have a different implementation
                return other + self
            return Bag([self, other])
        if isinstance(other, int):
            # Try to avoid awkward "d6 + -1" formatting.
            if other == 0:
                return self
            if other < 0:
                return self - abs(other)
        return self._apply_operator(add, other)

    @overload
    def __mul__(self, other: Die[F]) -> Die[F]:
        ...

    @overload
    def __mul__(self, other: F) -> Die[F]:
        ...

    def __mul__(self, other):
        if isinstance(other, int):
            if other == 1:
                return self
        return self._apply_operator(mul, other)

    def __rmul__(self, other: int) -> Die[F]:
        if other == 1:
            return self
        return Repeated(other, self)

    @overload
    def __sub__(self, other: Die[F]) -> Die[F]:
        ...

    @overload
    def __sub__(self, other: F) -> Die[F]:
        ...

    def __sub__(self, other):
        if isinstance(other, int):
            if other == 0:
                return self
            if other < 0:
                return self + abs(other)
        return self._apply_operator(sub, other)

    @overload
    def __truediv__(self, other: Die[F]) -> Die[F]:
        ...

    @overload
    def __truediv__(self, other: F) -> Die[F]:
        ...

    def __truediv__(self, other):
        return self._apply_operator(truediv, other)

    @overload
    def __floordiv__(self, other: Die[F]) -> Die[F]:
        ...

    @overload
    def __floordiv__(self, other: F) -> Die[F]:
        ...

    def __floordiv__(self, other):
        return self._apply_operator(facediv, other)

    @overload
    def __lt__(self, other: Die[F]) -> Die[F]:
        ...

    @overload
    def __lt__(self, other: F) -> Die[F]:
        ...

    def __lt__(self, other):
        return self._apply_operator(lt, other)

    @overload
    def __le__(self, other: Die[F]) -> Die[F]:
        ...

    @overload
    def __le__(self, other: F) -> Die[F]:
        ...

    def __le__(self, other):
        return self._apply_operator(le, other)

    @overload  # type: ignore
    def __eq__(self, other: Die[F]) -> Die[F]:
        ...

    @overload
    def __eq__(self, other: F) -> Die[F]:
        ...

    def __eq__(self, other):
        return self._apply_operator(eq, other)

    @overload  # type: ignore
    def __ne__(self, other: Die[F]) -> Die[F]:
        ...

    @overload
    def __ne__(self, other: F) -> Die[F]:
        ...

    def __ne__(self, other):
        return self._apply_operator(ne, other)

    @overload
    def __gt__(self, other: Die[F]) -> Die[F]:
        ...

    @overload
    def __gt__(self, other: F) -> Die[F]:
        ...

    def __gt__(self, other):
        return self._apply_operator(gt, other)

    @overload
    def __ge__(self, other: Die[F]) -> Die[F]:
        ...

    @overload
    def __ge__(self, other: F) -> Die[F]:
        ...

    def __ge__(self, other):
        return self._apply_operator(ge, other)

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
    def values(self) -> Slicer[F]:
        return Slicer(self)


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


@dataclasses.dataclass
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

    def get_precedence(self) -> int:
        return self.precedence

    def __str__(self):
        parts = [self.maybe_wrap(self.left), self.symbol, self.maybe_wrap(self.right)]
        return " ".join(parts)

    def get_faces(self) -> Faces[F_co]:
        return self.left.faces.combine(self.operator, self.right.faces)


@Operator(symbol="+", precedence=_ADD)
def add(left: SupportsAddF, right: SupportsAddF) -> SupportsAddF:
    return left + right


@Operator(symbol="-", precedence=_SUB)
def sub(left: SupportsSubF, right: SupportsSubF) -> SupportsSubF:
    return left - right


@Operator(symbol="*", precedence=_MUL)
def mul(left: SupportsMulF, right: SupportsMulF) -> SupportsMulF:
    return left * right


@Operator(symbol="/", precedence=_DIV)
def truediv(left: SupportsTrueDivF, right: SupportsTrueDivF) -> SupportsTrueDivF:
    return left / right


@Operator(symbol="//", precedence=_DIV)
def facediv(left: SupportsFaceDivF, right: SupportsFaceDivF) -> SupportsFaceDivF:
    quot, rem = divmod(left, right)
    return quot + (1 if rem else 0)


@Operator(symbol='<', precedence=_CMP)
def lt(left: SupportsLtF, right: SupportsLtF) -> SupportsLtF:
    return left < right


@Operator(symbol='<=', precedence=_CMP)
def le(left: SupportsLeF, right: SupportsLeF) -> SupportsLeF:
    return left <= right


@Operator(symbol='==', precedence=_CMP)
def eq(left: SupportsEqF, right: SupportsEqF) -> SupportsEqF:
    return left == right


@Operator(symbol='!=', precedence=_CMP)
def ne(left: SupportsNeF, right: SupportsNeF) -> SupportsNeF:
    return left != right


@Operator(symbol='>', precedence=_CMP)
def gt(left: SupportsGtF, right: SupportsGtF) -> SupportsGtF:
    return left > right


@Operator(symbol='>=', precedence=_CMP)
def ge(left: SupportsGeF, right: SupportsGeF) -> SupportsGeF:
    return left >= right


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
        return f"{self.number} x {self.maybe_wrap(self.die)}"

    def get_faces(self) -> Faces[F]:
        ret = self.die.faces
        for _ in range(self.number - 1):
            ret = ret.combine(lambda a, b: a + b, self.die.faces)
        return ret

    def get_contained(self) -> Iterable[Die[F]]:
        return self.number * [self.die]

    def __add__(self, other):
        if self << other:
            return dataclasses.replace(self, number=self.number + 1)
        return super().__add__(other)


@dataclasses.dataclass
class Bag(Die[F]):

    dice: Sequence[Die[F]]

    def get_precedence(self) -> int:
        return _ADD

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

    @overload
    def __add__(self, other: Die[F]) -> Die[F]:
        pass

    @overload
    def __add__(self, other: F) -> Die[F]:
        pass

    def __add__(self, other):
        if isinstance(other, int):
            return super().__add__(other)
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
class Item(Die[F]):

    die: Die[F]
    key: int

    def __str__(self) -> str:
        return f"{self.maybe_wrap(self.die)}[{self.key}]"

    def get_faces(self) -> Faces[F]:
        return self.die.slice_values(slice(self.key, self.key + 1)).faces.map(
            lambda v: v[0]
        )


@dataclasses.dataclass
class Slice(Die[Tuple[F]]):

    die: Die[F]
    key: slice

    def __str__(self) -> str:
        parts = [str(s) if s is not None else ""
                 for s in (self.key.start, self.key.stop)]
        if self.key.step is not None:
            parts.append(str(self.key.step))
        return f"{self.maybe_wrap(self.die)}[{':'.join(parts)}]"

    def get_faces(self) -> Faces[Tuple[F]]:
        return self.die.slice_values(self.key).faces


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


@dataclasses.dataclass
class MethodCall(Die[F]):

    receiver: Die
    func: Callable[..., Faces[F]]
    args: Sequence[Die[F]]
    kwargs: Dict[Any, Any]

    def __str__(self):
        if isinstance(self.receiver, OperatorCall) or isinstance(self.receiver, Bag):
            receiver = f"({self.receiver})"
        else:
            receiver = str(self.receiver)
        parts = [str(arg) for arg in self.args] + [
            f"{k}={v!r}" for k, v in self.kwargs.items()
        ]
        return f"{receiver}.{self.func.__name__}({', '.join(parts)})"

    def get_faces(self) -> Faces:
        return self.func(self.receiver, *self.args, **self.kwargs)


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


class Builder:

    def __call__(self, arg: int) -> DX:
        return DX(arg)

    def __getitem__(self, arg: Sequence[F]) -> Die[F]:
        return Seq(arg)

    def constant(self, arg: F) -> Die[F]:
        return Constant(arg)
