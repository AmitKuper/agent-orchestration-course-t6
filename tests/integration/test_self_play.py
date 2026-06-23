"""Integration test: one full self-play game, cop vs thief QTableActor."""

from game.game import Game

from actor_t6.qtable_actor import QTableActor


def test_full_game_no_illegal_moves() -> None:
    """A full game completes with no illegal moves and no exceptions."""
    cop = QTableActor(role="cop")
    thief = QTableActor(role="thief")
    cop.epsilon = 1.0
    thief.epsilon = 1.0

    game = Game.new(
        game_id="integration-test",
        grid_size=(5, 5),
        cop_pos=(0, 0),
        thief_pos=(4, 4),
        mechanics={"max_moves": 25, "max_barriers": 5},
    )

    done = False
    max_turns = 200  # guard against infinite loop
    turns = 0

    while not done and turns < max_turns:
        for role, backend in [("cop", cop), ("thief", thief)]:
            obs = game.get_state(role)
            action = backend.get_action(obs)

            assert action in obs.legal_moves, (
                f"{role} chose illegal action {action!r}; legal={obs.legal_moves}"
            )

            result = game.submit_action(role, action)
            backend.on_result(obs, action, result)

            if result.game_over:
                done = True
                break

        turns += 1

    assert done, "Game did not terminate within max_turns"


def test_full_game_terminates_with_winner() -> None:
    """Game always terminates with a winner string."""
    cop = QTableActor(role="cop")
    thief = QTableActor(role="thief")
    game = Game.new("test-2", (5, 5), (0, 0), (4, 4), {"max_moves": 25})

    winner = None
    for _ in range(500):
        for role, backend in [("cop", cop), ("thief", thief)]:
            obs = game.get_state(role)
            action = backend.get_action(obs)
            result = game.submit_action(role, action)
            backend.on_result(obs, action, result)
            if result.game_over:
                winner = result.winner
                break
        if winner:
            break

    assert winner in ("cop", "thief")
