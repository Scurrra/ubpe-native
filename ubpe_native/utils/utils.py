def copy[T](smth: str | tuple[T, ...] | list[T]) -> str | tuple[T, ...] | list[T]:
    """
    Universal function for copying. Supports strings, tuples, and lists.
    """
    return (
        smth
        if isinstance(smth, str)
        else tuple(smth)
        if isinstance(smth, tuple)
        else smth.copy()
    )


def join[T](
    *smth: str | tuple[T, ...] | list[T],
) -> str | tuple[T, ...] | list[T] | None:
    """
    Join a sequence of sequences (same type) into a single sequence. Supports strings, tuples, and lists.
    """
    if len(smth) == 0:
        return None

    eltype = type(smth[0])
    # ensure that the type of each argument is the same
    for i in range(1, len(smth)):
        # if not, return `None`
        if eltype is not type(smth[i]):
            return None

    # strings are simply joined
    if eltype is str:
        return "".join(smth)  # type: ignore

    # if arguments are not of type the function was created for
    if eltype is not tuple and eltype is not list:
        # just return a tuple from `*smth`
        return tuple(smth)  # type: ignore

    # for effective copying:
    # 1. find the length of result as sum of length of all the elements
    length = 0
    for i in range(len(smth)):
        length += len(smth[0])
    # 2. construct list of `None`s
    result = [None] * length
    # 3. `smth` by `smth` initialize `result`
    start = 0
    for i in range(len(smth)):
        result[start : (start + len(smth[i]))] = smth[i]  # type: ignore (lengths of a sublist and `smth[i]` are guaranteed to be the same)
        start += len(smth[i])
    # 4. keep `result` a list or convert it to a tuple
    return result if eltype is list else tuple(result)  # type: ignore (`None` was already return at this point)