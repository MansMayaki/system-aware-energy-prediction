# Pre-upload checklist

Before sharing the anonymous repository:

- Run `python scripts/validate_artifact.py`.
- Run `bash scripts/check_anonymity.sh`.
- Open the Anonymous GitHub mirror in an incognito/private browser.
- Confirm no author names, affiliations, email addresses, usernames, acknowledgments, or personal filesystem paths appear.
- Confirm the manuscript and repository describe the same released dataset version.
- Confirm the energy target name in the manuscript matches `energy_consumed_kWh` in `data/configurations.csv`.
- Confirm `power_correction` is documented; in the supplied release its only observed value is [1.0].
- Do not add `CITATION.cff` with author identities until the anonymous review is complete.
- After acceptance, create an attributed tagged release and archive it in a persistent repository (e.g. Zenodo) if desired.
