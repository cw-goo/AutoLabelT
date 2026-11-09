"""Update checks stay offline until an AutoLabel release feed is configured.

The upstream product's releases are not compatible AutoLabel application updates.
Model downloads use their separate registered sources.
"""


def check_for_updates_sync(timeout=10):
    return None


def check_for_updates_async(callback=None, timeout=10):
    return None
