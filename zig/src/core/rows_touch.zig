//! Wall-clock touches let the supervisor share an LRU with opaque native state files.
const std = @import("std");
const platform = @import("sys");
const rows = @import("rows_store.zig");
const pc = @import("prompt_cache.zig");
const chain = @import("rows_chain.zig");
const file = @import("rows_file.zig");

pub fn mark(r: *rows.Rows, i: usize) void {
    var path: [1200]u8 = undefined;
    const stem = std.fmt.bytesToHex(chain.stem(r.states.items[i].name), .lower);
    const p = file.path(&path, r.dir, &stem, ".used", false) catch return;
    writeWord(p, r.clock);
}

pub fn cached(r: *rows.Rows, e: *pc.Entry, starts: []const u32) void {
    const own = rows.startsBelow(r, r.gpa, starts, e.at) catch return;
    defer r.gpa.free(own);
    const name = chain.state(r.keying, e.tokens, e.decode_spans, own, e.at);
    const i = r.find(name) orelse return;
    var path: [1200]u8 = undefined;
    const stem = std.fmt.bytesToHex(chain.stem(name), .lower);
    const p = file.path(&path, r.dir, &stem, ".st", false) catch return;
    var opened = file.open(p, .state, name) catch {
        r.remove(i);
        for (r.queue.items) |q| if (q.entry == e) return;
        rows.write.queued(r, e, starts);
        return;
    };
    opened.close();
    r.touch(i);
}

pub fn readWord(p: [:0]const u8) u64 {
    const fd = platform.c.open(p, .{ .ACCMODE = .RDONLY }, @as(platform.c.mode_t, 0));
    if (fd < 0) return 0;
    defer _ = platform.c.close(fd);
    var w: u64 = 0;
    return if (platform.c.pread(fd, std.mem.asBytes(&w), 8, 0) == 8) w else 0;
}

pub fn writeWord(p: [:0]const u8, w: u64) void {
    var buf: [1200]u8 = undefined;
    const part = std.fmt.bufPrintSentinel(&buf, "{s}.part", .{p}, 0) catch return;
    const fd = platform.c.open(part, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true, .NOFOLLOW = true }, @as(platform.c.mode_t, 0o600));
    if (fd < 0) return;
    defer _ = platform.c.close(fd);
    if (platform.c.write(fd, std.mem.asBytes(&w), 8) != 8) {
        _ = platform.c.unlink(part);
        return;
    }
    _ = platform.c.rename(part, p);
}
