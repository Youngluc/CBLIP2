"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import sys

from omegaconf import OmegaConf

from ChineseBLIP2.common.registry import registry

from ChineseBLIP2.datasets.builders import *
from ChineseBLIP2.models import *
from ChineseBLIP2.processors import *
from ChineseBLIP2.tasks import *


root_dir = os.path.dirname(os.path.abspath(__file__))
default_cfg = OmegaConf.load(os.path.join(root_dir, "configs/default.yaml"))

registry.register_path("library_root", root_dir)
repo_root = os.path.join(root_dir, "..")
registry.register_path("repo_root", repo_root)
cache_root = os.path.join(repo_root, default_cfg.env.cache_root)
registry.register_path("cache_root", cache_root)
model_repo_root = default_cfg.env.model_repo_root
registry.register_path("model_repo_root", model_repo_root)
dataset_repo_root = default_cfg.env.dataset_repo_root
registry.register_path("dataset_repo_root", dataset_repo_root)

registry.register("MAX_INT", sys.maxsize)
registry.register("SPLIT_NAMES", ["train", "val", "test"])
