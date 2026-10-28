"""Landlock for one evaluation's worker process (README.md).

A kernel is model-written code. Its process may write only its scratch folder, device files (the
GPU's) and its own /proc entry (CUDA needs it), may open no TCP connection or listener, and
(Landlock ABI 6) may not signal
or reach any process outside its own sandbox, so it can't touch the server or another job.
Reading stays open: the image is read-only and holds no secrets. Landlock covers TCP only, so the
container also runs on a network with no way out (README.md); this is the second layer, inside
the container.
"""
import ctypes
import os

# Syscall numbers are the same on x86_64 and arm64.
_CREATE_RULESET, _ADD_RULE, _RESTRICT_SELF = 444, 445, 446
_CREATE_RULESET_VERSION = 1
_RULE_PATH_BENEATH = 1
_PR_SET_NO_NEW_PRIVS = 38

# Filesystem rights that write (reading and executing are not handled, so they stay open).
_WRITE_FILE, _REMOVE_DIR, _REMOVE_FILE = 1 << 1, 1 << 4, 1 << 5
_MAKE_CHAR, _MAKE_DIR, _MAKE_REG, _MAKE_SOCK = 1 << 6, 1 << 7, 1 << 8, 1 << 9
_MAKE_FIFO, _MAKE_BLOCK, _MAKE_SYM = 1 << 10, 1 << 11, 1 << 12
_REFER, _TRUNCATE = 1 << 13, 1 << 14              # ABI 2, ABI 3
_NET_BIND_TCP, _NET_CONNECT_TCP = 1 << 0, 1 << 1  # ABI 4
_SCOPE_ABSTRACT_UNIX, _SCOPE_SIGNAL = 1 << 0, 1 << 1  # ABI 6


class _PathBeneath(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]


_libc = ctypes.CDLL(None, use_errno=True)
_libc.syscall.restype = ctypes.c_long


def abi() -> int:
    """The kernel's Landlock ABI version, 0 when Landlock is off or blocked (seccomp)."""
    v = _libc.syscall(_CREATE_RULESET, None, ctypes.c_size_t(0), ctypes.c_uint32(_CREATE_RULESET_VERSION))
    return int(v) if v > 0 else 0


def restrict(writable: list[str]) -> dict:
    """Restrict this process (and every child it starts) for good. Raises OSError if it can't."""
    version = abi()
    if version < 1:
        raise OSError("Landlock is not available")
    fs = (_WRITE_FILE | _REMOVE_DIR | _REMOVE_FILE | _MAKE_CHAR | _MAKE_DIR | _MAKE_REG | _MAKE_SOCK
          | _MAKE_FIFO | _MAKE_BLOCK | _MAKE_SYM)
    if version >= 2:
        fs |= _REFER
    if version >= 3:
        fs |= _TRUNCATE
    attr = [fs]
    if version >= 4:
        attr.append(_NET_BIND_TCP | _NET_CONNECT_TCP)     # no rules: no TCP at all
    if version >= 6:
        attr.append(_SCOPE_ABSTRACT_UNIX | _SCOPE_SIGNAL)
    buf = (ctypes.c_uint64 * len(attr))(*attr)
    ruleset = _libc.syscall(_CREATE_RULESET, ctypes.byref(buf), ctypes.c_size_t(ctypes.sizeof(buf)),
                            ctypes.c_uint32(0))
    if ruleset < 0:
        raise OSError(ctypes.get_errno(), "landlock_create_ruleset")
    try:
        for path in writable:
            fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
            try:
                rule = _PathBeneath(fs, fd)
                if _libc.syscall(_ADD_RULE, ctypes.c_int(ruleset), ctypes.c_int(_RULE_PATH_BENEATH),
                                 ctypes.byref(rule), ctypes.c_uint32(0)) < 0:
                    raise OSError(ctypes.get_errno(), f"landlock_add_rule {path}")
            finally:
                os.close(fd)
        if _libc.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "PR_SET_NO_NEW_PRIVS")
        if _libc.syscall(_RESTRICT_SELF, ctypes.c_int(ruleset), ctypes.c_uint32(0)) < 0:
            raise OSError(ctypes.get_errno(), "landlock_restrict_self")
    finally:
        os.close(ruleset)
    return {"landlock_abi": version, "writable": writable, "tcp": "denied" if version >= 4 else "allowed",
            "signals_and_abstract_sockets": "own sandbox only" if version >= 6 else "not scoped"}
