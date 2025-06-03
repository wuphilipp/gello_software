# Copyright 2015 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ament_pep257.main import main
import pytest
from pathlib import Path


@pytest.mark.linter
@pytest.mark.pep257
def test_pep257():
    excluded_file = Path(__file__).resolve().parents[1] / "franka_gello_state_publisher/driver.py"
    rc = main(argv=[".", "test", "--exclude", str(excluded_file)])
    assert rc == 0, "Found code style errors / warnings"
