"""
Inngest client initialization.
"""

import os
import logging
import inngest
from .config import Config

inngest_client = inngest.Inngest(
    app_id="mirofish",
    is_production=not Config.DEBUG and bool(os.environ.get("INNGEST_SIGNING_KEY")),
    logger=logging.getLogger("mirofish.inngest")
)
