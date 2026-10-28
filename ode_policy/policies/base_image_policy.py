from typing import Dict

import torch
from ode_policy.utils.module_attr_mixin import ModuleAttrMixin
from ode_policy.utils.normalizer import LinearNormalizer


# Keys that travel inside obs_dict at inference but are NOT observations: they are
# flow-source signals the eval runner supplies alongside the real obs. They are
# absent from shape_meta, so the obs encoder ignores them — but the obs normalizer
# iterates whatever dict it is handed and raises KeyError on an unknown key, so
# every policy must strip them before normalizing. Defined once here so a new
# source cannot be added to the runner and silently break twelve policies.
NON_OBS_KEYS = ("past_action", "measured_delta")


class BaseImagePolicy(ModuleAttrMixin):
    # init accepts keyword argument shape_meta, see config/task/*_image.yaml

    def _normalize_obs(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Normalize only the real observation keys (see NON_OBS_KEYS)."""
        return self.normalizer.normalize(
            {k: v for k, v in obs_dict.items() if k not in NON_OBS_KEYS}
        )

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict:
            str: B,To,*
        return: B,Ta,Da
        """
        raise NotImplementedError()

    # reset state for stateful policies
    def reset(self):
        pass

    # ========== training ===========
    # no standard training interface except setting normalizer
    def set_normalizer(self, normalizer: LinearNormalizer):
        raise NotImplementedError()
