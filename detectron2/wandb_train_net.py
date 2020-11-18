#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Detection Training Script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import logging
import glob
import json
import os
import sys
from collections import OrderedDict
import torch

import detectron2.utils.comm as comm
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.engine.train_loop import SimpleTrainer
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.utils.events import CommonMetricPrinter, JSONWriter, TensorboardXWriter

import wandb
import wandb_detectron


class DefaultTrainNetTrainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can use the cleaner
    "SimpleTrainer", or write your own training loop. You can use
    "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                    ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "cityscapes_instance":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(
                    cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


class WandbTrainer(DefaultTrainNetTrainer):
    # def __init__(self, cfg):
    #     """
    #     Args:
    #         cfg (CfgNode):
    #     """
    #     logger = logging.getLogger("detectron2")
    #     # setup_logger is not called for d2
    #     if not logger.isEnabledFor(logging.INFO):
    #         setup_logger()
    #     # Assume these objects must be constructed in this order.
    #     model = self.build_model(cfg)
    #     optimizer = self.build_optimizer(cfg, model)
    #     data_loader = self.build_train_loader(cfg)

    #     # For training, wrap with DDP. But don't need this for inference.
    #     if comm.get_world_size() > 1:
    #         model = DistributedDataParallel(
    #             model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
    #         )

    #     # Note: we're not calling super here, because we want to skip DefaultTrainer's init.
    #     # TODO: Fix? To fix this we need to get rid of DefaultTrainer entirely. We could still
    #     #   delegate to it's methods, by assinging them to our methods.
    #     SimpleTrainer.__init__(self, model, data_loader, optimizer)

    #     self.scheduler = self.build_lr_scheduler(cfg, optimizer)
    #     # Assume no other objects need to be checkpointed.
    #     # We can later make it checkpoint the stateful hooks

    #     # Use wandb checkpointer instead of fvcore checkpointer. This checkpointer
    #     # automatically makes an input artifact reference for the loaded model
    #     self.checkpointer = wandb_detectron.WandbCheckpointer(
    #         # Assume you want to save checkpoints together with logs/statistics
    #         model,
    #         cfg.OUTPUT_DIR,
    #         optimizer=optimizer,
    #         scheduler=self.scheduler,
    #     )

    #     self.start_iter = 0
    #     self.max_iter = cfg.SOLVER.MAX_ITER
    #     self.cfg = cfg

    #     self.register_hooks(self.build_hooks())

    def build_writers(self):
        # Add wandb writer to save training metrics
        return [
            # It may not always print what you want to see, since it prints "common" metrics only.
            CommonMetricPrinter(self.max_iter),
            JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
            wandb_detectron.WandbWriter()
            # Don't use the tensorboard writer because it clears logged images, and so do we.
        ]

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.append(wandb_detectron.WandbModelSaveHook(
            self.cfg.OUTPUT_DIR,
            # TODO: get from config
            {'bbox.AP': wandb_detectron.UP_IS_BETTER},
            self.cfg.SOLVER.CHECKPOINT_PERIOD,
            self.cfg.TEST.EVAL_PERIOD,
            self.cfg.TEST.EVAL_PERIOD
        ))
        return hooks


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    wandb_detectron.wandb_register_artifact_datasets(cfg)

    if args.eval_only:
        run = wandb.init(config=cfg, job_type='eval')
        model = WandbTrainer.build_model(cfg)
        wandb_detectron.WandbCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = WandbTrainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(WandbTrainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)

        ds_artifact_name = wandb_detectron.remove_prefix(cfg.DATASETS.TEST[0],
                                                         wandb_detectron.WANDB_ARTIFACT_PREFIX)
        dataset_artifact = wandb.run.use_artifact(ds_artifact_name)
        datadir = dataset_artifact.download()

        metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
        if hasattr(metadata, 'thing_dataset_id_to_contiguous_id'):
            reverse_id_mapping = {
                v: k for k, v in metadata.thing_dataset_id_to_contiguous_id.items()
            }
            def get_original_class_id(x): return reverse_id_mapping[x]
        else:
            def get_original_class_id(x): return x

        eval_artifact = wandb.Artifact(
            type='result',
            name='run-%s-preds' % run.id,
            metadata=res)

        # eval_artifact.add_file(os.path.join(datadir, 'classes.json'))

        # TODO: this is a hack until we have have cross-artifact references
        # eval_artifact.add_file(os.path.join(datadir, 'dataset.table.json'))
        # eval_artifact.add(dataset_artifact.get("{}_table".format(ds_artifact_name)),)
        original_table = dataset_artifact.get("{}_table".join(ds_artifact_name))
        # import pdb; pdb.set_trace()
        example_preds = torch.load(
            os.path.join(cfg.OUTPUT_DIR, 'inference', 'instances_predictions.pth'))
        table = wandb.Table(['preds', 'id'])
        for example_pred in example_preds:
            id_str = str(example_pred['image_id'])
            image_path = os.path.join("images", "000000{}.jpg".format(id_str))
            boxes = []
            for instance in example_pred['instances']:
                box = instance['bbox']
                boxes.append({
                    'domain': 'pixel',
                    'position': {
                        'minX': box[0],
                        'maxX': box[0] + box[2],
                        'minY': box[1],
                        'maxY': box[1] + box[3]
                    },
                    'scores': {
                        'score': instance['score']
                    },
                    'class_id': get_original_class_id(instance['category_id'])
                })
            wandb_image = wandb.Image(dataset_artifact.get_path(image_path).download(),
                boxes={
                    'preds': {
                        'box_data': boxes
                    }
                })
            table.add_data(wandb_image, id_str)
        # eval_artifact.add(table, 'pred_table')
        eval_artifact.add(wandb.JoinedTable(original_table, table, "id"), "joined_prediction_table")
            # 'dataset.table.json', 'preds.table.json', 'path'),
            # 'preds.joined-table.json')
        wandb.run.log_artifact(eval_artifact)

        # for file in glob.glob(os.path.join(cfg.OUTPUT_DIR, 'inference', '*')):
        #     if not os.path.isfile(file):
        #         continue
        #     eval_artifact = wandb.Artifact(
        #         type='result',
        #         name='run-%s-%s' % (run.id, os.path.basename(file)),
        #         metadata=res)
        #     with eval_artifact.new_file('dataset.json') as f:
        #         # TODO: we should use the URI for whatever our input artifact
        #         #   ended up being, rather than what's passed in via test
        #         # TODO: This writes an array, because there can be more than one
        #         #   test dataset. How should we handle that case?
        #         # TODO: standardize how to do this.
        #         json.dump({'dataset_artifact': cfg.DATASETS.TEST}, f)
        #     eval_artifact.add_file(file)
        #     wandb.run.log_artifact(eval_artifact)
        wandb.run.log(res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop or subclassing the trainer.
    """
    # TODO: track --eval-only from args
    # TODO: only do this on main process?
    wandb.init(config=cfg, job_type='train')
    trainer = WandbTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(
                0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
