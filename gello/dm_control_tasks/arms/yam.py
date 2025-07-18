"""YAM composer class."""

from pathlib import Path
from typing import Union

from dm_control import mjcf

from gello.dm_control_tasks.arms.manipulator import Manipulator
from gello.dm_control_tasks.mjcf_utils import MENAGERIE_ROOT


class YAM(Manipulator):
    """YAM arm for dm_control simulation.

    This class uses all 8 joints from the MJCF model:
    - 6 arm joints (joint1-joint6)
    - 2 gripper joints (left_finger, right_finger)

    Note: The right_finger is passive and controlled by equality constraint.
    """

    XML = MENAGERIE_ROOT / "i2rt_yam" / "yam.xml"

    def _build(
        self,
        name: str = "YAM",
        xml_path: Union[str, Path] = XML,
    ) -> None:
        super()._build(name=name, xml_path=xml_path, gripper_xml_path=None)

    @property
    def flange(self) -> mjcf.Element:
        return self._mjcf_root.find("site", "grasp_site")
