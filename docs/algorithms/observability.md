# Mount Observability

Mount estimation is split between a reduced alignment estimator and the EKF runtime.

## What Is Observable From IMU/GNSS

GNSS velocity changes and raw IMU angular/specific-force signals make mount pitch and yaw strongly visible during ordinary acceleration, braking, and turning. Mount roll is weaker because many ground-vehicle motions can trade vehicle roll, road bank, and mount roll while producing similar body-frame measurements.

Nonholonomic constraints help the EKF enforce vehicle-frame motion consistency:

```text
v_y ~= 0
v_z ~= 0
```

Those constraints are defined in the vehicle frame. They affect the EKF through the current attitude and mount estimate, but the measurement still observes vehicle-frame velocity consistency, not mount roll as an isolated physical quantity. During banked-road motion, a wrong mount roll can be partially compensated by a different vehicle roll/attitude history, so low residuals do not prove unique mount-roll identification.

## Vehicle-Roll Prior

The runtime includes an optional soft vehicle-roll prior at eligible NHC epochs. Its default variance density is currently enabled in `RuntimeConfig` as `r_vehicle_roll_prior = 0.1`; setting it to `0` disables the update.

This prior is a flat-road assumption:

```text
vehicle_roll ~= 0
```

It can reduce mount-roll ambiguity on mostly flat driving, but it is not a sensor-derived proof that the road is flat. Sustained banked roads can violate the assumption, so roll-prior tuning should be evaluated against banked scenarios and field data.

## Alignment Versus EKF

Align is a reduced formulation with mount angles as its main states. It can seed the EKF well when the motion excites the mount, but it cannot fully separate mount roll from vehicle roll/bank by itself. The EKF has more states and a dynamic model, but it can still inherit a poor seed when the data does not identify the same decomposition.

The correct public claim is therefore narrower than "NHC observes mount roll": NHC and GNSS help constrain the full vehicle-motion solution; mount roll needs sufficiently informative motion, assumptions, or priors to become practically separable.
