"""
Example: use `incremental` to efficiently add files to an existing artifact.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import wandb

project = Path(__file__).stem
artifact_name = "incremental-artifact"

example_dir = Path("files")
example_dir.mkdir(exist_ok=True)

# Create an artifact with a single file.
file1 = example_dir / "one.txt"
file1.write_text("1")

with wandb.init(project=project, job_type="dataset") as run:
    # The original artifact can be created normally. (Does not need `incremental=True`)
    artifact = wandb.Artifact(artifact_name, type="dataset")
    artifact.add_file(file1)
    run.log_artifact(artifact)


# Create a new file and add it to the artifact; keeping the old file isn't necessary.
file1.unlink()
file2 = example_dir / "two.txt"
file2.write_text("2")

with wandb.init(project=project, job_type="dataset") as run:
    # To extend the existing artifact, it must be created with `wandb.Artifact` and the
    # same name, along with the `incremental=True` flag. `wandb.use_artifact()` can't
    # (currently) be used to extend an artifact.
    artifact = wandb.Artifact(artifact_name, type="dataset", incremental=True)
    artifact.add_dir(example_dir)  # `add_file` or `add_reference` can also be used.
    run.log_artifact(artifact)


# There is now a new version of the artifact that contains all three files.
with wandb.init(project=project, job_type="dataset") as run:
    artifact = wandb.use_artifact(artifact_name + ":latest")
    path = Path(artifact.download())
    print({str(p): p.read_text() for p in path.rglob("*")})


# (cleanup)
file2.unlink()
file3.unlink()
example_dir.rmdir()
