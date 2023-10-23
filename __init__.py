import os
import huggingface_hub.utils

RELAY_SERVER = os.environ.get("RELAY_SERVER", "https://relay.yimo.space")

def patch_url(url):
    if not url.startswith(RELAY_SERVER):
        url = os.path.join(RELAY_SERVER, url)
    return url

def patched_http_backoff(*args, **kwargs):
    if len(args) >= 2 and isinstance(args[1], str):
        args[1] = patch_url(args[1])
    elif "url" in kwargs and isinstance(kwargs["url"], str):
        kwargs["url"] = patch_url(kwargs["url"])
    return huggingface_hub.utils._http.http_backoff(*args, **kwargs)

huggingface_hub.utils.http_backoff = patched_http_backoff
