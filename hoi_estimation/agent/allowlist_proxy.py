#!/usr/bin/env python3
"""Network bridge for an agent container started with `--net --network none`.

host mode  : HTTP CONNECT proxy on a unix socket. Either allow-list mode (--allow host:port ...: only those are
             relayed) or block-list mode (--deny-suffix dom ...: everything except hosts ending in those domains is
             relayed). Every request (allowed or refused) is logged. The container's only route out is this socket.
inner mode : runs inside the container; listens on 127.0.0.1:<port> (TCP) and forwards each connection to the
             unix socket, so clients can use HTTPS_PROXY=http://127.0.0.1:<port>.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import time


async def pipe(r, w):
    try:
        while data := await r.read(65536):
            w.write(data)
            await w.drain()
    except (ConnectionError, asyncio.IncompleteReadError):
        pass
    finally:
        try:
            w.close()
        except Exception:
            pass


def permitted(target, allow, deny):
    host = target.rsplit(":", 1)[0].lower().rstrip(".")
    if deny is not None:
        return not any(host == d or host.endswith("." + d) for d in deny)
    return target in allow


async def host_main(sock, allow, log, deny=None):
    lf = open(log, "a", buffering=1)

    async def handle(r, w):
        head = await r.readuntil(b"\r\n\r\n")
        line = head.split(b"\r\n", 1)[0].decode(errors="replace")
        parts = line.split()
        ok = len(parts) >= 2 and parts[0] == "CONNECT" and permitted(parts[1], allow, deny)
        lf.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {'ALLOW' if ok else 'DENY '} {line}\n")
        if not ok:
            w.write(b"HTTP/1.1 403 Forbidden\r\n\r\n")
            await w.drain()
            w.close()
            return
        host, port = parts[1].rsplit(":", 1)
        try:
            ur, uw = await asyncio.open_connection(host, int(port))
        except OSError as e:
            lf.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} FAIL  {line} ({e})\n")
            w.write(b"HTTP/1.1 502 Bad Gateway\r\n\r\n")
            await w.drain()
            w.close()
            return
        w.write(b"HTTP/1.1 200 Connection established\r\n\r\n")
        await w.drain()
        await asyncio.gather(pipe(r, uw), pipe(ur, w))

    if os.path.exists(sock):
        os.unlink(sock)
    srv = await asyncio.start_unix_server(handle, path=sock)
    async with srv:
        await srv.serve_forever()


async def inner_main(sock, port):
    async def handle(r, w):
        ur, uw = await asyncio.open_unix_connection(sock)
        await asyncio.gather(pipe(r, uw), pipe(ur, w))

    srv = await asyncio.start_server(handle, "127.0.0.1", port)
    async with srv:
        await srv.serve_forever()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["host", "inner"])
    ap.add_argument("--sock", required=True)
    ap.add_argument("--allow", nargs="*", default=["www.micuapi.ai:443"])
    ap.add_argument("--deny-suffix", nargs="*", default=None, help="block-list mode: refuse these domains (and subdomains)")
    ap.add_argument("--log", default="/dev/null")
    ap.add_argument("--port", type=int, default=18080)
    a = ap.parse_args()
    asyncio.run(host_main(a.sock, set(a.allow), a.log, a.deny_suffix) if a.mode == "host" else inner_main(a.sock, a.port))
