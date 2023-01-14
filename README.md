# R2RISE
Explaining imitation learning through frames.

# Install baseslines
Install existing basesline file first.

# Make directories
mkdir ./checkpoint

mkdir -p ./image/BC/beamrider/test/

mkdir -p ./image/BC/breakout/test/

mkdir -p ./image/GAIL/beamrider/test/

mkdir -p ./image/GAIL/breakout/test/

mkdir -p ./output_npz/beamrider/frames

mkdir -p ./output_npz/breakout/frames

# Run R2RISE

python BC.py --env_name breakout

python GAIL.py --env_name breakout
