import dataclasses
import operator
from typing import (
    Callable,
    Dict,
    Generator,
    Iterable,
    Optional,
)

from rocks_fall import dice


def generate_ruin(
    max_ruin: int = 6,
    max_rolls: int = 100,
    stop_at: float = 0.99,
    starting_values: Optional[Iterable[int]] = None,
    compare: Callable[[int, int], bool] = operator.gt,
    dark_die: Optional[dice.Die[int]] = None,
    light_die: Optional[dice.Die[int]] = None,
    num_light_dice: int = 1,
) -> Generator[Dict, None, None]:

    if starting_values is None:
        starting_values = range(1, max_ruin)
    if dark_die is None:
        dark_die = dice.DX(max_ruin)
    assert isinstance(dark_die, dice.Die)
    if light_die is None:
        light_die = dark_die
    assert isinstance(light_die, dice.Die)

    # You could also use tuples for state, e.g.:
    #
    # RuinRoll = Tuple[int, bool]  # dark value, is highest
    # State = Tuple[int, int, int]  # starting, current, roll number
    #
    # I think using dataclasses is clearer, since the field names are part of the
    # code instead of just part of the documentation.

    @dataclasses.dataclass(frozen=True, order=True)
    class RuinRoll:
        dark_value: int
        is_highest: bool = True

    @dataclasses.dataclass(frozen=True, order=True)
    class State:
        ruin: int
        starting_ruin: int
        roll_number: int = 0

    def advance(a: State, b: RuinRoll) -> State:
        increase = compare(b.dark_value, a.ruin) and b.is_highest
        return dataclasses.replace(
            a,
            roll_number=a.roll_number + 1,
            ruin=min(a.ruin + increase, max_ruin),
        )

    def time_to_stop(state_die: dice.Die[State]) -> bool:
        return any(v.ruin == max_ruin and w >= stop_at for v, w in state_die.faces)

    if num_light_dice:
        light_dice = num_light_dice * light_die
        ruin_die = dark_die.combine(
            lambda a, b: RuinRoll(a, a >= b), light_dice.highest()
        ).named(f"ruin(dark={dark_die}, light={light_dice})")
    else:
        ruin_die = dark_die.map(RuinRoll).named(f"ruin(dark={dark_die})")
    print(f'Ruin die: {ruin_die}')

    def ruin_results(starting_ruin: int) -> Generator[Dict, None, None]:
        state_die: dice.Die[State] = dice.Constant(
            State(ruin=starting_ruin, starting_ruin=starting_ruin)
        )
        for roll in max_rolls * [ruin_die]:
            state_die = state_die.combine(advance, roll)
            for f, w in state_die.faces:
                yield dict(dataclasses.asdict(f), probability=float(w))
            if time_to_stop(state_die):
                break

    for starting_ruin in starting_values:
        for r in ruin_results(starting_ruin):
            yield r


for row in list(generate_ruin())[:5]:
    print(row)
