#!/usr/bin/env python3
import logging

from soloair_control_service import main


if __name__ == "__main__":
    logging.warning(
        "soloair_av_service.py is now a compatibility entry. "
        "Use scripts/start_soloair_stack.sh to start system RTSP plus conda control."
    )
    main()
