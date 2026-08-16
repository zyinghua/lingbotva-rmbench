#!/usr/bin/env bash

# Replacement for RoboTwin/script/_install.sh. Copy this file there, then run
# it from the RoboTwin repository root.
set -euo pipefail

if [[ ! -f script/requirements.txt || ! -d envs ]]; then
    echo "Error: run this installer from the RoboTwin repository root." >&2
    exit 1
fi

echo "Installing the necessary packages ..."
python -m pip install -r script/requirements.txt

echo "Installing pytorch3d ..."
# cd third_party/pytorch3d_simplified
# pip install -e .
# cd ../..
python -m pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" --no-build-isolation

echo "Adjusting code in sapien/wrapper/urdf_loader.py ..."
# location of sapien, like "~/.conda/envs/RoboTwin/lib/python3.10/site-packages/sapien"
SAPIEN_LOCATION=$(python -m pip show sapien | grep 'Location' | awk '{print $2}')/sapien
# Adjust some code in wrapper/urdf_loader.py
URDF_LOADER=$SAPIEN_LOCATION/wrapper/urdf_loader.py
# ----------- before -----------
# 667         with open(urdf_file, "r") as f:
# 668             urdf_string = f.read()
# 669 
# 670         if srdf_file is None:
# 671             srdf_file = urdf_file[:-4] + "srdf"
# 672         if os.path.isfile(srdf_file):
# 673             with open(srdf_file, "r") as f:
# 674                 self.ignore_pairs = self.parse_srdf(f.read())
# ----------- after  -----------
# 667         with open(urdf_file, "r", encoding="utf-8") as f:
# 668             urdf_string = f.read()
# 669 
# 670         if srdf_file is None:
# 671             srdf_file = urdf_file[:-4] + ".srdf"
# 672         if os.path.isfile(srdf_file):
# 673             with open(srdf_file, "r", encoding="utf-8") as f:
# 674                 self.ignore_pairs = self.parse_srdf(f.read())
sed -i -E 's/("r")(\))( as)/\1, encoding="utf-8") as/g' "$URDF_LOADER"


echo "Adjusting code in mplib/planner.py ..."
# location of mplib, like "~/.conda/envs/RoboTwin/lib/python3.10/site-packages/mplib"
MPLIB_LOCATION=$(python -m pip show mplib | grep 'Location' | awk '{print $2}')/mplib

# Adjust some code in planner.py
# ----------- before -----------
# 807             if np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit:
# 808                 return {"status": "screw plan failed"}
# ----------- after  ----------- 
# 807             if np.linalg.norm(delta_twist) < 1e-4 or not within_joint_limit:
# 808                 return {"status": "screw plan failed"}
PLANNER=$MPLIB_LOCATION/planner.py
sed -i -E 's/(if np.linalg.norm\(delta_twist\) < 1e-4 )(or collide )(or not within_joint_limit:)/\1\3/g' "$PLANNER"

echo "Installing Curobo ..."

# CuRobo v0.7.8 uses setuptools-scm while preparing editable-install
# metadata. Ubuntu's setuptools 59.6 is incompatible with setuptools-scm 8.1,
# so install the known-compatible build pair before invoking pip. Warp is a
# CuRobo runtime dependency; pin the version used by RoboTwin with v0.7.8.
python -m pip install --upgrade \
    setuptools==69.5.1 \
    setuptools-scm==8.1.0 \
    warp-lang==1.12.0

CUROBO_TAG="v0.7.8"
CUROBO_COMMIT="d64c4b005459db10c5dd867d8b30a87d5bda9bdb"

cd envs

# The RoboTwin checkout is commonly bind-mounted, so envs/curobo can persist
# across containers. Reuse it only when it is the exact expected checkout;
# never overwrite or delete an unexpected directory.
if [[ ! -e curobo ]]; then
    git clone --branch "$CUROBO_TAG" --depth 1 \
        https://github.com/NVlabs/curobo.git curobo
elif [[ ! -d curobo/.git ]]; then
    echo "Error: envs/curobo exists but is not a Git checkout." >&2
    exit 1
fi

ACTUAL_CUROBO_COMMIT=$(git -C curobo rev-parse HEAD)
if [[ "$ACTUAL_CUROBO_COMMIT" != "$CUROBO_COMMIT" ]]; then
    echo "Error: expected CuRobo $CUROBO_TAG ($CUROBO_COMMIT), found $ACTUAL_CUROBO_COMMIT." >&2
    exit 1
fi

cd curobo
python -m pip install -e . --no-build-isolation
cd ../..

echo "Installation basic environment complete!"
echo -e "You need to:"
echo -e "    1. \033[34m\033[1m(Important!)\033[0m Download assets from huggingface."
echo -e "    2. Install requirements for running baselines. (Optional)"
echo "See INSTALLATION.md for more instructions."
