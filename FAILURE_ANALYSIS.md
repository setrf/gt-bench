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

## Implication

For this benchmark, the next targeted improvement would be more tie-heavy training examples or a balanced evaluation slice by number of equilibria. I would not expand the game-theory scope until this residual edge case is handled.
