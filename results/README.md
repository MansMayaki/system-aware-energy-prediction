# Reference outputs

`reference_outputs/` contains outputs produced by the revision-analysis scripts from the supplied analysis-ready dataset.

The directory mirrors the analysis stages:

- `00_audit`: dataset counts and aggregation diagnostics;
- `01_grouped_validation`: configuration-family-disjoint validation and conformal intervals;
- `01b_workload_family_validation`: stricter workload-family validation;
- `02_ablations`: linear ablations and nonlinear baselines;
- `03_prerun`: nested pre-run efficiency prediction, feature importance, conformal intervals, and scale-shift tests;
- `04_diagnostics`: correlation, VIF, condition-number, cluster-SE, and memory-proxy robustness analyses;
- `05_hardware_scaling`: hardware-specific fits, leave-one-hardware-out transfer, interactions, and strict matched scaling.

The original absolute input path was removed from the released audit metadata to preserve anonymity.
