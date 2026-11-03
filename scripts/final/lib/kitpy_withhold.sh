#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# kitpy_withhold.sh: the kit python ($KIT_PYTHON) with all arguments passed through, except that the kit packer's one
# clean-reference stdout line ("locally computed: ...") is replaced by a marker before the caller sees or stores it.
# Used for Tier 2 and Tier 3, whose selections never look at a clean score (team rule). The kit is not modified; the
# exit status is python's.
set -o pipefail
"${KIT_PYTHON:?KIT_PYTHON is not set}" "$@" | sed -u 's/^locally computed:.*$/locally computed: [withheld by kitpy_withhold.sh: never shown or stored]/'
