# yinying edit this file from deepmind alphafold repo

"""Docker launch script for PRIME docker image."""

import os
import signal

import docker
from absl import app
from absl import flags
from docker import types
from typing import Tuple

flags.DEFINE_string("fasta", None, "Path to a specific FASTA filename.")
flags.DEFINE_string("mutant", None, "Path to a specific mutant filename")
flags.DEFINE_string("save", "./prime_predictions/test.csv", "Saving csv path.")
flags.DEFINE_string("checkpoint", './checkpoints/prime_base.pt', "Path to a specific mutant filename")

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
    _ROOT_MOUNT_DIRECTORY = os.path.abspath("/tmp/")
    os.makedirs(_ROOT_MOUNT_DIRECTORY, exist_ok=True)

def _create_mount(mount_name: str, path: str, read_only=True) -> Tuple[types.Mount, str]:
    """Create a mount point for each file and directory used by the model."""
    path = os.path.abspath(path)
    target_path = os.path.join(_ROOT_MOUNT_DIRECTORY, mount_name)

    if os.path.isdir(path):
        source_path = path
        mounted_path = target_path
    else:
        source_path = os.path.dirname(path)
        mounted_path = os.path.join(target_path, os.path.basename(path))
    if not os.path.exists(source_path):
        os.makedirs(source_path)
    print('Mounting %s -> %s', source_path, target_path)
    mount = types.Mount(target=str(target_path), source=str(source_path),
                        type='bind', read_only=read_only)
    return mount, str(mounted_path)

def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    mounts = []
    command_args = []

    if FLAGS.fasta:
        fasta = os.path.abspath(FLAGS.fasta)
        mount_fasta, mounted_fasta=_create_mount(mount_name='fasta', path=fasta,read_only=True)
        mounts.append(mount_fasta)
        command_args.append(f"--fasta={mounted_fasta}")
    
    if FLAGS.mutant:
        mutant = os.path.abspath(FLAGS.mutant)
        mount_mutant,mounted_mutant=_create_mount(mount_name='mutant', path=mutant, read_only=True)
        mounts.append(mount_mutant)
        command_args.append(f"--mutant={mounted_mutant}")

    save = os.path.abspath(FLAGS.save)

    os.makedirs(os.path.dirname(save), exist_ok=True)
    mount_save,mounted_save=_create_mount(mount_name='save',path=save,read_only=False)
    mounts.append(mount_save)
    command_args.append(f"--save={mounted_save}")

    checkpoint = os.path.abspath(FLAGS.checkpoint)

    os.makedirs(os.path.dirname(checkpoint), exist_ok=True)
    mount_checkpoint, mounted_checkpoint=_create_mount(mount_name='checkpoint', path=checkpoint, read_only=False)
    mounts.append(mount_checkpoint)
    command_args.append(f"--checkpoint={mounted_checkpoint}")

    print(command_args)

    client = docker.from_env()
    network=client.networks.create("network1", driver="bridge")

    container = client.containers.run(
        image=FLAGS.docker_image_name,
        command=command_args,
        remove=True,
        detach=True,
        mounts=mounts,
        user=FLAGS.docker_user,
    )
    network.connect(container)

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
