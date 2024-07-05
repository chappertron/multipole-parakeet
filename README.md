## About this package
[![Powered by MDAnalysis](https://img.shields.io/badge/powered%20by-MDAnalysis-orange.svg?logoWidth=16&logo=data:image/x-icon;base64,AAABAAEAEBAAAAEAIAAoBAAAFgAAACgAAAAQAAAAIAAAAAEAIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAJD+XwCY/fEAkf3uAJf97wGT/a+HfHaoiIWE7n9/f+6Hh4fvgICAjwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACT/yYAlP//AJ///wCg//8JjvOchXly1oaGhv+Ghob/j4+P/39/f3IAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAJH8aQCY/8wAkv2kfY+elJ6al/yVlZX7iIiI8H9/f7h/f38UAAAAAAAAAAAAAAAAAAAAAAAAAAB/f38egYF/noqAebF8gYaagnx3oFpUUtZpaWr/WFhY8zo6OmT///8BAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgICAn46Ojv+Hh4b/jouJ/4iGhfcAAADnAAAA/wAAAP8AAADIAAAAAwCj/zIAnf2VAJD/PAAAAAAAAAAAAAAAAICAgNGHh4f/gICA/4SEhP+Xl5f/AwMD/wAAAP8AAAD/AAAA/wAAAB8Aov9/ALr//wCS/Z0AAAAAAAAAAAAAAACBgYGOjo6O/4mJif+Pj4//iYmJ/wAAAOAAAAD+AAAA/wAAAP8AAABhAP7+FgCi/38Axf4fAAAAAAAAAAAAAAAAiIiID4GBgYKCgoKogoB+fYSEgZhgYGDZXl5e/m9vb/9ISEjpEBAQxw8AAFQAAAAAAAAANQAAADcAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAjo6Mb5iYmP+cnJz/jY2N95CQkO4pKSn/AAAA7gAAAP0AAAD7AAAAhgAAAAEAAAAAAAAAAACL/gsAkv2uAJX/QQAAAAB9fX3egoKC/4CAgP+NjY3/c3Nz+wAAAP8AAAD/AAAA/wAAAPUAAAAcAAAAAAAAAAAAnP4NAJL9rgCR/0YAAAAAfX19w4ODg/98fHz/i4uL/4qKivwAAAD/AAAA/wAAAP8AAAD1AAAAGwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAALGxsVyqqqr/mpqa/6mpqf9KSUn/AAAA5QAAAPkAAAD5AAAAhQAAAAEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADkUFBSuZ2dn/3V1df8uLi7bAAAATgBGfyQAAAA2AAAAMwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAB0AAADoAAAA/wAAAP8AAAD/AAAAWgC3/2AAnv3eAJ/+dgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA9AAAA/wAAAP8AAAD/AAAA/wAKDzEAnP3WAKn//wCS/OgAf/8MAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIQAAANwAAADtAAAA7QAAAMAAABUMAJn9gwCe/e0Aj/2LAP//AQAAAAAAAAAA)](https://www.mdanalysis.org)

A python package containing an MDAnalysis class to calculate charge, dipolar and quadrupolar density profiles from simulation trajectories. 
Also includes scripts (in the bin folder) to use this class from the command line and a class for calculating per molecule electrostatic properties.

If you use this package, please link to this page, cite our paper (Chapman and Bresme 2022, unpublished) and [cite the papers](https://www.mdanalysis.org/pages/citations/) of the MDAnalaysis authors

## Requirements and Dependencies

### Python Packages:
- setuptools
- MDAnalysis(https://www.mdanalysis.org/)
- numpy and matplotlib
- (optional) pip


## How to install
*These instructions should work on Linux and MacOS. You may need to do additional steps to add the scripts to your path on Windows.*

To install, download the repo and change directory to it. Next either run:
- `python setup.py install` or
- `pip install ./`

The latter might be preferred as it is easier to uninstall (with `pip uninstall multipole`) and will try to automatically download dependancies, if available on pip. When using the first option you may want to use an enviroment manager, such as the one in conda.

Installing with previous instructions will add the script `calc_multipoles_dens` to your python installation's bin folder (e.g. for anaconda, `~/anaconda3/bin/`). This is typically in your path, so you can use `calc_multipoles_dens` directly from the command line.

## Running 
The command line utility `calc_multipole_dens` has only one mandatory option, a topology file. This however will only give you the charge densities for the frame represented by that topology with the default settings. It is very important to change the settings for your needs.

TODO: Update this to use correct args
The help message for `calc_multipole_dens` is as follows:
```bash
usage: calc_multipoles_dens [-h] [-f Trajectory] [-v] [-m] [--water_model {tip4p05,spce,tip3p,opc}]
                            [-b BEGIN] [-e END] [-M M_NAME] [-H H_NAME [H_NAME ...]] [-w BIN_WIDTH]
                            [-c] [--check_unwrapped] [--types_or_names {None,type,name}]
                            topfile

Calculate the bins of the charge, dipole and quadrapole. Currently only implemented for water..
This can be done for just a single frame, or an entire trajectory

positional arguments:
  topfile               The topology file name

options:
  -h, --help            show this help message and exit
  -f Trajectory, --trajfile Trajectory
  -v, --verbose         Increase verbosity
  -m, --inmem           Load Trajectory Into memory
  --water_model {tip4p05,spce,tip3p,opc}
                        calculate the mutipole moments for the specified water model. Applies some
                        pre-processing such as assigning charges if not already. Future versions
                        will make None default, so non water systems can be used.
  -b BEGIN, --begin BEGIN
  -e END, --end END
  -M M_NAME, --M_name M_NAME
                        the atom name or type to use for the centre of the molecule for binning and
                        multipole moment calculation. Should use O for tip3p/spc styles and dummy
                        atom for tip4p styles.
  -H H_NAME [H_NAME ...], --H_name H_NAME [H_NAME ...]
  -w BIN_WIDTH, --bin_width BIN_WIDTH
                        The target width of the bins to use, in Angstrom. Overridden if number of
                        bins is set instead
  -c, --coord_centre    if used, binned coordinate is centre of the bin, else it is the left edge
  --check_unwrapped
  --types_or_names {None,type,name}
                        Specify whether hydrogen/centre choice is in reference to the atom type or
                        name. By default tries to find names
```

The option `-f` is where you specify the trajectory file you wish to use. 
It is important that you set the water model to match that of the system you are using, with the `--water_model` option. Currently only the TIP4P/2005, SPC/E, TIP3P and OPC models work.

You need to set `-M` and `-H` to the name or type of the negatively charged atom (oxygen or the dummy atom) and the hydrogen atoms, respectively. Whether name or type is used depends on the `--types_or_names` option.

Because this package is based on MD analysis, it should accept a wide range of trajectory and topology file formats, however, not all have been tested to work with this script. The topology and trajcetory formats MD analysis can accept are listed [here](https://userguide.mdanalysis.org/stable/formats/index.html).


## Understanding the results
Running the script will output a file (that can be read with `np.load_txt`, for example) containing the charge densites, the multipole densities, the molecule density and the orientations of the molecules.

To calculate the the electostatic field and potentials, and dipolar and quadrupolar contributions, you need to integrate the charge/multipolar densities. The package [thermotar](https://github.com/chappertron/thermotar) has some tools for doing this.

## Limitations
- Currently these scripts work with only a few water models.
- ~~If your water model uses dummy atoms, but the trajectories do not include the positions of the dummy atoms (such as the case with the output from LAMMPS), you must add these back in. This script does not yet do this whilst reading the trajectory, however you can generate a trajectory that re-adds the atoms using our [dummify](https://github.com/chappertron/dummify) package. ~~
- The location of the origin for calculating the dipole and quadrupole of each molecule must be on the negatively charged atom (real or dummy). The choice in orign has no effect on the dipole moments, but can affect the quadrupole contribution. For water the location of the negative charge is a good choice.
- Electrostatic fields/potentials must be calculated from the average density profiles in post processing, no support for calculating for each frame yet.
- No sub-averaging for estimating errors in one trajectory is performed. Use independant repeats and average and calculate the standard error of the potential profiles from these repeats.
- Binning is only avalible in 1D currently.
- Charges that change in time are unsupported.

## Changes From 0.0.1 to 0.1
- Can calculate the position of the dummy atoms on the fly using the `-d` option.
- Performs unwrapping of water molecules on the fly (needed for multipole and dummy calculation). Specify with `-u`.
- Supports the presence of charges not due to water molecules. Binned in the total charge and independently. 
<!-- - Charge densities are now in $e\ \mathrm{\AA^{-3}}$ instead of $\mathrm{C\ \AA^{-3}}$. -->
- Now requires [`numba`]() for fast calculations of the quadrupole, dipole moments, box unwrapping and dummy atom position.
  - Due to JIT compilation, This may be slower for short trajectories/single frames. In the future this may be optional.
- Now requires [`fast_histogram`]() for fast calculation of the charge binning. This was chosen because the `numba` implementation of `np.histogram` does not support weights, which are essential for binning.

## Warning about DCD Trajectories
If using `MDAnalysis` version `2.4.X` for `X` < 3,  there is a bug when reading DCD large trajectories that causes crashing. This is fixed in newer versions, so make sure to use the latest version or use `2.3` or earlier. 

### TODO

- [x] Unwrap trajectories using oxygen hydrogen distance, or MD ANAlysis residues.
- [ ] Add the option to only calculate the charge profiles.
- [ ] Add calculations  of the potential and field at the end of the simulation.
- [ ] Update the README.md for the command line arguments.
- [ ] Calculate charge densities in e/AA^3. Add option for old behaviour.