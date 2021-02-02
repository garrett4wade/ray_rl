from ray.util.queue import Empty


def drain_ray_queue(queue_obj, n_sentinel=0, guard_sentinel=False):
    """Empty a multiprocessing queue object, with options to protect against
    the delay between ``queue.put()`` and ``queue.get()``.  Returns a list of
    the queue contents.

    With ``n_sentinel=0``, simply call ``queue.get(block=False)`` until
    ``queue.Empty`` exception (which can still happen slightly *after* another
    process called ``queue.put()``).

    With ``n_sentinel>1``, call ``queue.get()`` until `n_sentinel` ``None``
    objects have been returned (marking that each ``put()`` process has finished).

    With ``guard_sentinel=True`` (need ``n_sentinel=0``), stops if a ``None``
    is retrieved, and puts it back into the queue, so it can do a blocking
    drain later with ``n_sentinel>1``.
    """
    contents = list()
    if n_sentinel > 0:  # Block until this many None (sentinels) received.
        sentinel_counter = 0
        while True:
            obj = queue_obj.get()
            if obj is None:
                sentinel_counter += 1
                if sentinel_counter >= n_sentinel:
                    return contents
            else:
                contents.append(obj)
    while True:  # Non-blocking, beware of delay between put() and get().
        try:
            obj = queue_obj.get(block=False)
        except Empty:
            return contents
        if guard_sentinel and obj is None:
            # Restore sentinel, intend to do blocking drain later.
            queue_obj.put(None)
            return contents
        elif obj is not None:  # Ignore sentinel.
            contents.append(obj)
