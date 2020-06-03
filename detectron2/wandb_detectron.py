import glob
import logging
import json
import os
import shutil
import sys

from detectron2.utils.events import EventWriter, get_event_storage
from detectron2.data.datasets.coco import load_coco_json
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine.hooks import HookBase
from detectron2 import checkpoint
from fvcore.common.file_io import PathManager

import wandb
from wandb import wandb_run
import torch

WANDB_ARTIFACT_PREFIX = 'wandb-artifact://'

DETECTRON2_PREFIX = 'detectron2://'

logger = logging.getLogger(__name__)

def remove_prefix(from_string, prefix):
    return from_string[len(prefix):]

def register_wandb_dataset(artifact_uri):
    if DatasetCatalog._REGISTERED.get(artifact_uri):
        return

    def use_artifact():
        if wandb.run is None:
            logger.error('Can\'t use wandb dataset outside of run, please call wandb.init()')
            raise ValueError('wandb not initialized')
        artifact_alias = remove_prefix(artifact_uri, WANDB_ARTIFACT_PREFIX)
        artifact = wandb.run.use_artifact(type='dataset', name=artifact_alias)
        if artifact is None:
            logger.error('Error: W&B dataset artifact doesn\'t exist')
            raise ValueError('Artifact doesn\'t exist')

        if (not artifact.metadata or not
                artifact.metadata.get('format') or not
                artifact.metadata.get('format').get('type')):
            raise ValueError('Artifact missing metadata.format.type')
        datadir = artifact.download()
        format_type = artifact.metadata['format']['type']
        if format_type == 'coco_dataset_json':
            json_file_path = os.path.join(datadir, 'annotations.json')
            image_root = os.path.join(datadir, 'images/')

            metadata = {}
            metadata_path = os.path.join(datadir, 'detectron2_metadata.json')
            if os.path.isfile(metadata_path):
                metadata = json.load(open(metadata_path))
            # Careful! json doesn't support integer keys, which detectron uses for this mapping
            if 'thing_dataset_id_to_contiguous_id' in metadata:
                metadata['thing_dataset_id_to_contiguous_id'] = {
                    int(did): cid for did, cid in metadata['thing_dataset_id_to_contiguous_id'].items()}
            MetadataCatalog.get(artifact_uri).set(
                json_file=json_file_path, image_root=image_root, evaluator_type="coco", **metadata
            )
            return load_coco_json(json_file_path, image_root, artifact_alias)
        else:
            raise ValueError('Unknown artifact format_type')

    # TODO: we can use the public API to get the artifact at register time, then
    # call run.use_artifact() on that at fetch time.
    DatasetCatalog.register(artifact_uri, use_artifact)

def wandb_register_artifact_datasets(cfg):
    DATASETS = cfg.DATASETS
    all_datasets = DATASETS.TRAIN + DATASETS.TEST
    for ds_name in all_datasets:
        if ds_name.startswith(WANDB_ARTIFACT_PREFIX):
            register_wandb_dataset(ds_name)

class WandbWriter(EventWriter):
    """
    Write all scalars to a tensorboard file.
    """

    def __init__(self, window_size: int = 20, **kwargs):
        """
        Args:
            log_dir (str): the directory to save the output events
            window_size (int): the scalars will be median-smoothed by this window size

            kwargs: other arguments passed to `wandb.init(...)`
        """
        self._window_size = window_size

    def write(self):
        if wandb.run is None:
            # wandb not initialized
            return

        storage = get_event_storage()
        log_dict = {}
        for k, v in storage.latest_with_smoothing_hint(self._window_size).items():
            log_dict[k] = v

        if len(storage.vis_data) >= 1:
            # TODO: mask_head logs images with indexes in the name like (23), so it
            # generates a lot of keys per step. We should fix that. Might need to
            # parse the image name, or send a patch to detectron
            for img_name, img, step_num in storage.vis_data:
                img = torch.tensor(img)
                log_dict[img_name] = wandb.Image(img.permute(1, 2, 0).cpu().numpy())
            storage.clear_images()

        wandb.log(log_dict, step=storage.iter)


class WandbCheckpointer(checkpoint.DetectionCheckpointer):
    def save(self, name: str, **kwargs) -> None:
        super().save(name, **kwargs)


    def load(self, path: str, checkpointables = None) -> object:
        if path.startswith(WANDB_ARTIFACT_PREFIX):
            artifact_alias = remove_prefix(path, WANDB_ARTIFACT_PREFIX)
            artifact = wandb.run.use_artifact(type='model', name=artifact_alias)
            if artifact is None:
                raise ValueError('W&B model artifact doesn\'t exist')
            datadir = artifact.download()
            files = glob.glob(os.path.join(datadir, '*.pth'))
            if len(files) > 1:
                raise ValueError('Expected a single .pth file in model artifact')
            # transform path to local path
            path = files[0]
        else:
            name = path
            if path.startswith(DETECTRON2_PREFIX):
                name = 'Detectron model zoo - %s' % remove_prefix(path, DETECTRON2_PREFIX)
            name = name.replace(':', '_').replace('/', '_')

            # pull the file using fvcore PathManager. It's going to get pulled again
            # by the checkpointer, so this is free.
            local_path = PathManager.get_local_path(path)
            # TODO: This is not uploading local_path for some reason.
            artifact = wandb.Artifact(type='model', name=name, metadata={
                'reference': path,
                'format': {'type': 'detectron_model'}
            })
            artifact.add_file(local_path)
            wandb.run.use_artifact(artifact)

        return super().load(path, checkpointables)

UP_IS_BETTER = 'up'
DOWN_IS_BETTER = 'down'

class WandbModelSaveHook(HookBase):
    def __init__(self, output_dir, metric_directions, checkpoint_period, eval_period, save_period):
        # TODO: should we take checkpointer and evaluator hooks as arguments, so we can
        #     walk through to the checkpointer to figure out file names for example?
        self._output_dir = output_dir
        if eval_period / checkpoint_period != int(eval_period / checkpoint_period):
            raise ValueError('eval_period is not a multiple of checkpoint_period')
        self._checkpoint_period = checkpoint_period
        self._eval_period = eval_period
        # TODO Are config values none if not defined?
        if (save_period is not None
                and save_period / checkpoint_period != int(save_period / checkpoint_period)):
            raise ValueError('save_period is not a multiple of checkpoint_period')
        self._save_period = save_period
        if len(metric_directions) == 0:
            raise ValueError('you must pass at least one metric direction')
        self._metric_directions = metric_directions
        for metric, direction in self._metric_directions.items():
            if direction != UP_IS_BETTER and direction != DOWN_IS_BETTER:
                raise ValueError('metric_direction values must be UP_IS_BETTER or DOWN_IS_BETTER')

        self._best_metrics = {}
        self._best_steps = {}
        self._saved_steps = {}

    def _get_metric_from_eval_results(self, eval_results, key):
        components = key.split('.')
        for comp in components:
            if comp not in eval_results:
                raise ValueError('eval results don\'t contain metric: %s' % metric)
            eval_results = eval_results[comp]
        if type(eval_results) != float:
            raise ValueError('metric value most be float for key: %s' % key)
        return eval_results

    def _metric_alias(self, metric):
        return ('best-%s' % metric).replace('/', '_').replace('\\', '_')
    
    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if not is_final and next_iter % self._checkpoint_period != 0:
            # no checkpoint on this step
            return
        if not is_final and next_iter % self._eval_period != 0:
            # no eval on this step
            return

        # Assume this exists on trainer. detectron2's DefaultTrainer does this.
        last_eval_results = self.trainer._last_eval_results

        # TODO: we're assuming the checkpointer output locations
        last_checkpoint_filename = open(os.path.join(self._output_dir, 'last_checkpoint')).read()
        checkpoint_file = os.path.join(self._output_dir, last_checkpoint_filename)

        improved_metrics = []
        for metric, direction in self._metric_directions.items():
            last_metric_val = self._get_metric_from_eval_results(last_eval_results, metric)
            best_metric_val = self._best_metrics.get(metric)
            if (best_metric_val is None or
                    direction == UP_IS_BETTER and last_metric_val > best_metric_val or
                    direction == DOWN_IS_BETTER and last_jetric_val < best_metric_val):
                improved_metrics.append(metric)
                self._best_metrics[metric] = last_metric_val

                model_metric_dir = os.path.join(self._output_dir, self._metric_alias(metric))
                os.makedirs(model_metric_dir, exist_ok=True)

                shutil.copy(checkpoint_file, os.path.join(model_metric_dir, 'model.pth'))
                json.dump(last_eval_results, open(os.path.join(model_metric_dir, 'metrics.json'), 'w'))

                metric_inference_dir = os.path.join(model_metric_dir, 'inference') 
                os.makedirs(metric_inference_dir, exist_ok=True)
                for inference_file in glob.glob(os.path.join(self._output_dir, 'inference', '*')):
                    shutil.copy(inference_file, metric_inference_dir)

        for metric in improved_metrics:
            self._best_steps[metric] = self.trainer.iter

        if is_final or next_iter % self._save_period == 0:
            # If this is a save step, save now
            # TODO: create run edge and log checkpoint artifact
            print('SAVING WITH IMPROVED', improved_metrics)
            artifact = wandb.Artifact(
                type='model',
                name='Trained by run - %s' % wandb.run.id,
                metadata={
                    'format': {'type': 'detectron_model'},
                    'training': {
                        'log_step': self.trainer.iter,
                        'metrics': last_eval_results
                    }
                })
            artifact.add_file(checkpoint_file, 'model.pth')
            wandb.run.log_artifact(artifact,
                aliases=[self._metric_alias(m) for m in improved_metrics])

            # Create an evaluation run that uses this artifact as input and generates
            # eval results.
            # Doesn't work right now. You can't create a run this way.
            # eval_run = wandb_run.Run(job_type='eval')
            # # eval_run.summary.update(last_eval_results)
            # eval_run.save()
            # eval_run.use_artifact(type='model', name=artifact.digest)
            # for file in glob.glob(os.path.join(self._output_dir, 'inference', '*')):
            #     eval_artifact = eval_run.new_artifact(
            #         type='result', name='Run - %s - %s' % (eval_run.id, os.path.basename(file)))
            #     eval_artifact.add_file(file)
            #     eval_artifact.save()

            for metric in improved_metrics:
                self._saved_steps[metric] = self.trainer.iter

    def after_train(self):
        for metric in self._metric_directions:
            if metric not in self._best_steps:
                # we never checkpointed this metric
                continue
            best_step = self._best_steps[metric]
            if metric not in self._saved_steps or self._saved_steps[metric] != best_step:
                print('SAVING FOR UNSAVED best', metric)
                artifact = wandb.Artifact(
                    type='model',
                    name='Trained by run - %s' % wandb.run.id,
                    metadata={
                        'format': {'type': 'detectron_model'},
                        'log_step': best_step
                        # TODO: We need last_eval_results, so we need to save that per best
                    })
                # TODO: This may call save more than once for the same step. Will
                # the backend logic correctly merge the aliases?
                artifact.add_file(checkpoint_file, 'model.pth')
                wandb.run.log_artifact(artifact,
                    aliases=[self._metric_alias(m) for m in improved_metrics])