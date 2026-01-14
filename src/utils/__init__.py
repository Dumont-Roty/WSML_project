"""Shared utilities for scraping and ML."""

# Provide a compatibility shim for pickled models that reference ml.src.identity_hasher
import sys
from . import identity_hasher

# Register under legacy path so pickled references work
sys.modules['ml.src.identity_hasher'] = identity_hasher
sys.modules['ml'] = sys.modules.get('ml') or type(sys)('ml')
if not hasattr(sys.modules['ml'], 'src'):
    sys.modules['ml'].src = type(sys)('src')
sys.modules['ml'].src.identity_hasher = identity_hasher
