#!/usr/bin/env bash
set -euo pipefail

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
package_dir=$(dirname -- "$script_dir")
environment_name="step-crosswalk-release"

cd "$package_dir"
conda env create --file environment/environment.yml
conda run --name "$environment_name" Rscript environment/install_r_packages.R
conda run --name "$environment_name" Rscript environment/validate_runtime.R
conda run --name "$environment_name" python -m unittest discover -s tests -v
