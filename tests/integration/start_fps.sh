#!/bin/bash
source ~/tmp/claude/setup_damd195_offline.sh >/dev/null 2>&1
cd ~/devel/ics/pfi/ics_fpsActor
exec python -m ics.fpsActor.main
