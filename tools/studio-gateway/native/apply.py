#!/usr/bin/env python3
"""Apply native session snapshot RPCs to the pinned Studio native source checkout.

No builds, model changes, service restarts, or credential operations are performed.
The patch is intentionally guarded by exact source anchors and can be re-applied.
"""
from pathlib import Path
import argparse
import shutil


def replace(path, old, new):
    text = path.read_text()
    if new in text:
        return
    if text.count(old) != 1:
        raise ValueError(f"Unexpected source at {path}: expected one anchor, got {text.count(old)}")
    path.write_text(text.replace(old, new))


def apply(root):
    source = Path(__file__).parent / 'payloads'
    for name, dst in [('session_file.zig', 'core'), ('session_file_test.zig', 'core'), ('session_routes.zig', 'server')]:
        shutil.copyfile(source / name, root / 'zig/src' / dst / name)
    p = root / 'zig/src/core/prompt_cache.zig'
    replace(p, '    active_rewind_at: u32 = 0,', '    // Explicit gateway sessions: checkpoint, drafter, executable, numeric probe and machine.\n    session_identity: ?u64 = null,\n    // An imported state must match the first resumed request, with no cold fallback.\n    session_required: ?Saved = null,\n    active_rewind_at: u32 = 0,')
    replace(p, '        if (e == null) s.counts.misses += 1;', '        if (s.session_required) |required| if (e == null or e.?.saved != required) return error.SessionPrefixMismatch;\n        if (e == null) s.counts.misses += 1;')
    replace(p, '        s.clock += 1;\n        e.used = s.clock;\n        e.branched', '        if (s.session_required == e.saved) s.session_required = null;\n        s.clock += 1;\n        e.used = s.clock;\n        e.branched')
    p = root / 'zig/src/core/rows_file.zig'
    replace(p, '    if (t.payload + t.meta + @sizeOf(Trailer) != @as(u64, @intCast(end))) return error.RowsRead;',
        '    const content_bytes = std.math.add(u64, t.payload, t.meta) catch return error.RowsRead;\n    const file_bytes = std.math.add(u64, content_bytes, @sizeOf(Trailer)) catch return error.RowsRead;\n    if (file_bytes != @as(u64, @intCast(end))) return error.RowsRead;')
    p = root / 'zig/src/core/engine_api.zig'
    text = p.read_text()
    if 'pub const session_file =' not in text:
        p.write_text(text + '\npub const session_file = @import("session_file.zig");\n')
    p = root / 'zig/src/core/rows_ask.zig'
    replace(p, 'pub const Op = enum { where, load, forget };', 'pub const Op = enum { where, load, forget, session_save, session_load, session_clear };\npub fn sessionOp(op: Op) bool { return op == .session_save or op == .session_load or op == .session_clear; }')
    replace(p, '    done: std.atomic.Value(bool)', '    session_path: [:0]const u8 = "",\n    session_receipt: @import("session_file.zig").Receipt = .{},\n    session_error: ?anyerror = null,\n    done: std.atomic.Value(bool)')
    replace(p, '        const s = store orelse return;', '        const s = store orelse { if (sessionOp(a.op)) a.session_error = error.SessionsDisabled; return; };\n        if (sessionOp(a.op)) {\n            a.session_receipt = (switch (a.op) {\n                .session_save => @import("session_file.zig").save(s, a.session_path),\n                .session_load => @import("session_file.zig").load(s, a.session_path),\n                .session_clear => @import("session_file.zig").clear(s),\n                else => unreachable,\n            }) catch |err| { a.session_error = err; return; };\n            return;\n        }')
    replace(p, '            .where => {},', '            .session_save, .session_load, .session_clear => unreachable,\n            .where => {},')
    p = root / 'zig/src/core/lane_host_rows.zig'
    replace(p, '    for (list) |a| a.answer(h.cache, .{ .spans = h.decoded_rows }); // keyed as its lookups are', '''    for (list) |a| {
        if (@import("rows_ask.zig").sessionOp(a.op)) {
            h.lock();
            const busy = h.queued.items.len > 0 or h.admitted.items.len > 0 or h.filling.items.len > 0 or h.core.activeCount() > 0;
            if (busy) {
                if (!h.disk.push(h.gpa, a)) { a.session_error = error.OutOfMemory; a.done.store(true, .release); }
                h.unlock();
                continue;
            }
            h.unlock();
        }
        a.answer(h.cache, .{ .spans = h.decoded_rows });
    }''')
    p = root / 'zig/src/native/qwen27_disk.zig'
    replace(p, '    store.disk = r;\n    std.log.info', '''    if (@import("sys").c.getenv("TF_SESSION_ROOT") != null) {
        const build_raw = @import("sys").c.getenv("TF_SESSION_BUILD_ID") orelse return error.SessionBuildIdMissing;
        const build = std.mem.span(build_raw);
        if (build.len != 64) return error.SessionBuildIdInvalid;
        for (build) |c| if (!std.ascii.isHex(c)) return error.SessionBuildIdInvalid;
        var identity_hash = std.hash.Wyhash.init(0x73657373696f6e31);
        identity_hash.update(std.mem.asBytes(&r.keying.identity));
        identity_hash.update(build);
        store.session_identity = identity_hash.final();
        store.most = 1;
        r.deinit();
        std.log.info("qwen27: explicit gateway sessions, one RAM snapshot, no automatic disk writes", .{});
        return;
    }
    store.disk = r;
    std.log.info''')
    p = root / 'zig/src/native/qwen27_slots.zig'
    replace(p, '            } else |_| {\n                s.reuse_failed = true;', '            } else |err| {\n                if (@import("sys").c.getenv("TF_SESSION_ROOT") != null) return err;\n                s.reuse_failed = true;')
    p = root / 'zig/src/server/routes.zig'
    replace(p, 'const imprint = @import("imprint_routes.zig");', 'const imprint = @import("imprint_routes.zig");\nconst sessions = @import("session_routes.zig");')
    replace(p, 'fn post(srv: *Server, conn: *Conn, a: Allocator) void {\n    const route = auth.routePath(conn.path);', 'fn post(srv: *Server, conn: *Conn, a: Allocator) void {\n    const route = auth.routePath(conn.path);\n    if (std.mem.startsWith(u8, route, sessions.prefix)) return sessions.post(srv, conn, a, route);')
    p = root / 'zig/src/server/imprint_routes.zig'
    replace(p, '        .forget => "imprint.forget",', '        .forget => "imprint.forget",\n        .session_save, .session_load, .session_clear => unreachable,')
    print(f'Applied native gateway session snapshot patch to {root}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    apply(parser.parse_args().root.resolve())
