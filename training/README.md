## Quickstart

### Setup overlay image

```
# open folder where you want to setup overlay
cp -rp /scratch/work/public/overlay-fs-ext3/overlay-25GB-500K.ext3.gz .
gunzip overlay-25GB-500K.ext3.gz
```
#### Allocate GPU to install GPU specific dependencies
```
srun --pty -c 4 --mem=16GB --time=00:30:00 --gres=gpu:rtx8000:1 /bin/bash
```
#### Run the overlay image
```
singularity exec --nv --overlay /scratch/sg7761/robomaster/yolonas/overlay-25GB-500K.ext3 /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash
```
#### Install conda
```
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p /ext3/miniconda3
# rm Miniconda3-latest-Linux-x86_64.sh # if you don't need this file any longer
```
#### Create script to source conda
Use `vi` to edit `/ext3/env.sh` and paste the following:
```
#!/bin/bash

source /ext3/miniconda3/etc/profile.d/conda.sh
export PATH=/ext3/miniconda3/bin:$PATH
export PYTHONPATH=/ext3/miniconda3/bin:$PATH
```
Save the file and run `source /ext3/env.sh`
#### Verify
Your terminal should look like this
```
Singularity> conda
usage: conda [-h] [-v] [--no-plugins] [-V] COMMAND ...

conda is a tool for managing and deploying applications, environments and packages.

options:
  -h, --help          Show this help message and exit.
  -v, --verbose       Can be used multiple times. Once for detailed output, twice for INFO logging, thrice for DEBUG logging, four times for TRACE logging.
  --no-plugins        Disable all plugins that are not built into conda.
  -V, --version       Show the conda version number and exit.

commands:
  The following built-in and plugins subcommands are available.

  COMMAND
    activate          Activate a conda environment.
    clean             Remove unused packages and caches.
    compare           Compare packages between conda environments.
    config            Modify configuration values in .condarc.
    content-trust     Signing and verification tools for Conda
    create            Create a new conda environment from a list of specified packages.
    deactivate        Deactivate the current active conda environment.
    doctor            Display a health report for your environment.
    info              Display information about current conda install.
    init              Initialize conda for shell interaction.
    install           Install a list of packages into a specified conda environment.
    list              List installed packages in a conda environment.
    notices           Retrieve latest channel notifications.
    package           Create low-level conda packages. (EXPERIMENTAL)
    remove (uninstall)
                      Remove a list of packages from a specified conda environment.
    rename            Rename an existing environment.
    repoquery         Advanced search for repodata.
    run               Run an executable in a conda environment.
    search            Search for packages and display associated information using the MatchSpec format.
    update (upgrade)  Update conda packages to the latest compatible version.
```
Create conda environment and use normally like https://github.com/naseemap47/YOLO-NAS

### Install and run dependencies - ON TERMINAL
Repositories like https://github.com/naseemap47/YOLO-NAS usually come with instructions. Following them usually gives best results. 

**NOTE** while installing dependencies you want your file to have a `rw` (read-write) permission but while executing code you want it to have `ro` (read-only) permission. Do NOT try updating the overlay file in `ro` mode as the changes wont be save in the right location. 

#### Allocate GPU to install GPU specific dependencies
You may vary the time and resources according to your needs
```
srun --pty -c 4 --mem=16GB --time=00:30:00 --gres=gpu:rtx8000:1 /bin/bash
```
#### Run the overlay image
```
singularity exec --nv --overlay /scratch/sg7761/robomaster/yolonas/overlay-25GB-500K.ext3:rw /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash
```
#### Activate conda
```
source /ext3/env.sh
conda activate <env-name> #yolo-nas for this example
```

Now you can run any python file for quick prototyping and making sure your code works

### Run dependencies - ASYNCHRONOUSLY
Suppose you are training a large model that takes hours to train and you cannot be the terminal open all the time. Use `SBATCH` files to specify your system configuration and  commands to run. SLURM will automatically allocate resources and run the commands when possible. An example of this can be found in `train_nas.SBATCH`. All communication happpens using emails. 

To start a `SBATCH` job, run:
```
sbatch <file-name> #train_nas.SBATCH in our case
```

To monitor a job:
```
watch squeue -u <netid> #enter your net id here
```

**NOTE** make sure to define the `.err` and `.out` file with unqiue names so that its easier to debug your problems.