# yinying edit this file from deepmind alphafold repo

"""Docker launch script for PRIME docker image."""

import os
import pathlib
import signal

import docker
from absl import app
from absl import flags
from docker import types
from typing import Tuple

flags.DEFINE_string("fasta", None, "Path to a specific FASTA filename.")
flags.DEFINE_string("mutant", None, "Path to a specific mutant filename")
flags.DEFINE_string("save", "./prime_predictions", "Saving directory path.")

flags.DEFINE_string(
    "docker_image_name", "prime-lianglab", "Name of the Pythia Docker image."
)

flags.DEFINE_string(
    "docker_user",
    f"{os.geteuid()}:{os.getegid()}",
    "UID:GID with which to run the Docker container. The output directories "
    "will be owned by this user:group. By default, this is the current user. "
    "Valid options are: uid or uid:gid, non-numeric values are not recognised "
    "by Docker unless that user has been created within the container.",
)

FLAGS = flags.FLAGS

try:
    _ROOT_MOUNT_DIRECTORY = f"/home/{os.getlogin()}"
except:
    _ROOT_MOUNT_DIRECTORY = pathlib.Path("/tmp/").resolve()
    os.makedirs(_ROOT_MOUNT_DIRECTORY, exist_ok=True)


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    mounts = []
    command_args = []

    if FLAGS.fasta:
        fasta = pathlib.Path(FLAGS.fasta).resolve()
        input_target_fasta_path = os.path.join(_ROOT_MOUNT_DIRECTORY, "fasta", os.path.basename(fasta))
        mounts.append(types.Mount(input_target_fasta_path, str(fasta), type="bind"))
        command_args.append(f"--fasta={input_target_fasta_path}")
    
    if FLAGS.mutant:
        mutant = pathlib.Path(FLAGS.mutant).resolve()
        input_target_mutant_path = os.path.join(_ROOT_MOUNT_DIRECTORY, "mutant", os.path.basename(mutant))
        mounts.append(types.Mount(input_target_mutant_path, str(mutant), type="bind"))
        command_args.append(f"--mutant={input_target_mutant_path}")

    save = pathlib.Path(FLAGS.save).resolve()

    os.makedirs(os.path.dirname(save), exist_ok=True)
    output_target_path = os.path.join(_ROOT_MOUNT_DIRECTORY, "output",os.path.basename(save))
    mounts.append(types.Mount(output_target_path, str(save), type="bind"))
    command_args.append(f"--save={output_target_path}")


    print(command_args)

    client = docker.from_env()

    container = client.containers.run(
        image=FLAGS.docker_image_name,
        command=command_args,
        remove=True,
        detach=True,
        mounts=mounts,
        user=FLAGS.docker_user,
    )

    # Add signal handler to ensure CTRL+C also stops the running container.
    signal.signal(signal.SIGINT, lambda unused_sig, unused_frame: container.kill())

    for line in container.logs(stream=True):
        print(line.strip().decode("utf-8"))


if __name__ == "__main__":
    flags.mark_flags_as_required([
        'fasta',
        'mutant',
    ])
    app.run(main)
