
from collections import defaultdict

import argparse
import os

from smartsim import Experiment

def main(args):

    # Get system-specific settings
    exp = Experiment("online-training", launcher="slurm")

    # Create the objects for the mock simulation
    if args.fortran_sim:
        rs_sim = exp.create_run_settings("build/mock_fortran_simulation")
    else:
        rs_sim = exp.create_run_settings("build/mock_cpp_simulation")
    rs_sim.set_tasks(4)
    rs_sim.set_nodes(1)
    model = exp.create_model("mock_simulation", rs_sim)
    model.attach_generator_files(to_symlink=["data"])

    # Create the objects for the sampler
    rs_sampler = exp.create_run_settings("python", exe_args="sampler.py")
    rs_sampler.set_tasks(1)
    rs_sampler.set_nodes(1)
    sampler = exp.create_model("sampler", rs_sampler)
    sampler.attach_generator_files(to_symlink=["sampler.py"])

    # Create the objects for the trainer
    rs_trainer = exp.create_run_settings("python", exe_args="trainer.py")
    rs_trainer.set_tasks(1)
    rs_trainer.set_nodes(1)
    trainer = exp.create_model("trainer", rs_trainer)
    trainer.attach_generator_files(to_symlink=["trainer.py"])

    # Create and configure the database
    db = exp.create_database(interface="bond0")

    try:
        exp.start(db)
        exp.generate(model, sampler, trainer, overwrite=True)
        exp.start(model, sampler, trainer, block=True)
    finally:
        exp.stop(model, sampler, trainer, db)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train data on the fly")
    parser.add_argument(
        "--fortran-sim",
        action="store_true",
        help="Use the Fortran-based mock simulation"
    )
    args = parser.parse_args()
    main(args)
