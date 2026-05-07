# Failure Analysis

The best 5000-example fine-tuned checkpoint made 2 errors on the canonical 500-example test set and 3 errors on the independent 1000-example confirmation set.

## Pattern

The remaining failures are tie-heavy games where the model over-predicts extra equilibria.

In these cases, the model often identifies a true equilibrium but adds one extra cell that is not actually a mutual best response. This suggests the residual weakness is not basic payoff parsing; it is precise conjunction of both players' best-response sets under ties.

## Canonical Test Failures

- `example_005625`
  - Gold: `(D, R)` and `(U, R)`
  - Prediction: `(U, R)`, `(D, L)`, and `(D, R)`
- `example_005990`
  - Gold: `(U, L)` and `(U, R)`
  - Prediction: `(U, L)`, `(U, R)`, and `(D, L)`

## Confirmation Failures

The confirmation run had the same qualitative failure mode: over-predicting extra equilibria in tie-heavy cases.

## Balanced Stress Set

I added a balanced 250-example stress set with 50 games each containing 0, 1, 2, 3, and 4 pure-strategy equilibria.

On this set, the baseline solved 204 of 250 examples. Its weakest bucket was zero-equilibrium games: 16 of 50 correct. The 5000-example fine-tuned checkpoint solved all 250 examples, including all 50 zero-equilibrium games and all 50 four-equilibrium tie cases.

This makes the failure story sharper: the base model often wants every 2x2 game to have at least one pure equilibrium, while the fine-tuned model has learned the exact mutual-best-response rule much more reliably.

## Implication

For this benchmark, the targeted improvement is now implemented as an adversarial prompt SFT pipeline. It adds compact-pair, JSON-like, answer-only, minimal-matrix, and standard-table examples while balancing zero-equilibrium and tie-heavy cases.

The balanced stress and prompt-robustness sets should remain regression tests. I would not expand the game-theory scope until the adversarial checkpoint is evaluated and the prompt-format failure mode is either reduced or clearly documented.
