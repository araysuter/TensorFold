//! CPU-only round trips use the existing fake multi-part KV/recurrent/ring family, not an implementation mirror.
const std = @import("std");
const sys = @import("sys");
const sf = @import("session_file.zig");
const pc = @import("prompt_cache.zig");
const Fake = @import("rows_fake.zig").Fake;
const Kept = @import("rows_fake.zig").Kept;
const rules: pc.Rules = .{ .lookahead = 1, .min_gap = 1, .min_prompt = 0, .chunk = 128 };

fn path(buf: []u8, tag: []const u8) [:0]const u8 {
    return std.fmt.bufPrintSentinel(buf, "/tmp/tf-session-{s}-{d}.tfs", .{ tag, sys.c.getpid() }, 0) catch unreachable;
}
fn populate(s: *pc.Store, f: *Fake, toks: []const u32) !void {
    f.prompt = toks;
    f.at = @intCast(toks.len - 1);
    try std.testing.expect(s.keep(toks, f.at, null, &.{}, f.spans));
}

test "two budgeted session checkpoints retain a rewind when the assistant suffix changes" {
    const a = std.testing.allocator;
    const toks = try a.alloc(u32, 6001);
    defer a.free(toks);
    for (toks, 0..) |*t, i| t.* = @intCast(31 + i * 17 % 10007);
    const next = try a.dupe(u32, toks);
    defer a.free(next);
    next[5995] += 1; // a canonical assistant rendering differs after the shared history
    for ([_]usize{ 1, 2, 2 }, [_]bool{ false, false, true }) |most, one_state_budget| {
        var f: Fake = .{ .gpa = a, .prompt = toks };
        var s = pc.Store.init(a, f.snapshots(), rules, 1 << 30);
        defer s.deinit();
        s.most = most;
        s.active_rewind_at = 5900;
        if (one_state_budget) s.budget = s.family.vtable.bytes(s.family.ptr, 6000);
        f.at = 5900;
        try std.testing.expect(s.keep(toks, 5900, null, &.{}, &.{}));
        f.at = 6000;
        try std.testing.expect(s.keep(toks, 6000, null, &.{}, &.{}));
        try std.testing.expect(s.held <= s.budget);
        try std.testing.expect(s.entries.items.len <= most);
        f.prompt = next;
        const plan = try s.begin(a, next, 6000, &.{}, &.{}, null, &.{});
        defer a.free(plan.marks);
        try std.testing.expectEqual(@as(u32, if (most == 2 and !one_state_budget) 5900 else 0), plan.from);
        if (plan.from > 0) try std.testing.expect(f.restored_ok);
    }
}

test "session snapshot restores full multi-part image and token/mode prefix exactly with one slot" {
    const a = std.testing.allocator;
    var buf: [128]u8 = undefined;
    const p = path(&buf, "roundtrip");
    defer _ = sys.c.unlink(p);
    const toks = try a.alloc(u32, 1501);
    defer a.free(toks);
    for (toks, 0..) |*t, i| t.* = @intCast(31 + i * 17 % 10007);
    var f: Fake = .{ .gpa = a, .spans = &.{.{ 300, 600 }} };
    var s = pc.Store.init(a, f.snapshots(), rules, 1 << 30);
    defer s.deinit();
    s.session_identity = 99;
    s.most = 1;
    try populate(&s, &f, toks);
    const original: *Kept = @ptrCast(@alignCast(s.entries.items[0].saved));
    const expected = try a.dupe(u8, original.img);
    defer a.free(expected);
    const receipt = try sf.save(&s, p);
    try std.testing.expectEqual(@as(u32, 1500), receipt.tokens);
    _ = try sf.clear(&s);
    try std.testing.expectEqual(@as(usize, 0), f.live);
    const restored = try sf.load(&s, p);
    try std.testing.expectEqual(receipt, restored);
    const actual: *Kept = @ptrCast(@alignCast(s.entries.items[0].saved));
    try std.testing.expectEqualSlices(u8, expected, actual.img);
    try std.testing.expectEqualSlices(u32, toks, s.entries.items[0].tokens);
    const plan = try s.begin(a, toks, 1500, &.{}, &.{}, null, f.spans);
    defer a.free(plan.marks);
    try std.testing.expectEqual(@as(u32, 1500), plan.from);
    try std.testing.expect(f.restored_ok);
    try std.testing.expect(s.session_required == null);
    try std.testing.expectEqual(@as(usize, 1), f.live);
}

test "session import rejects wrong model, wrong prefix, corrupt bytes and metadata before visible state" {
    const a = std.testing.allocator;
    var buf: [128]u8 = undefined;
    const p = path(&buf, "bad");
    defer _ = sys.c.unlink(p);
    const toks = [_]u32{ 10, 20, 30, 40, 50, 60 };
    var f: Fake = .{ .gpa = a };
    var s = pc.Store.init(a, f.snapshots(), rules, 1 << 30);
    defer s.deinit();
    s.session_identity = 100;
    s.most = 1;
    try populate(&s, &f, &toks);
    _ = try sf.save(&s, p);
    try std.testing.expectError(error.SessionMustClear, sf.load(&s, p));
    _ = try sf.clear(&s);
    s.session_identity = 101;
    try std.testing.expectError(error.RowsRead, sf.load(&s, p));
    try std.testing.expectEqual(@as(usize, 0), f.live);
    s.session_identity = 100;
    _ = try sf.load(&s, p);
    var other = toks;
    other[0] = 11;
    try std.testing.expectError(error.SessionPrefixMismatch, s.begin(a, &other, 5, &.{}, &.{}, null, &.{}));
    _ = try sf.clear(&s);
    const fd = sys.c.open(p, .{ .ACCMODE = .RDWR, .NOFOLLOW = true }, @as(sys.c.mode_t, 0));
    try std.testing.expect(fd >= 0);
    defer _ = sys.c.close(fd);
    try std.testing.expectEqual(@as(isize, 1), sys.c.pwrite(fd, "Q", 1, 17));
    try std.testing.expectError(error.SessionHash, sf.load(&s, p));
    try std.testing.expectEqual(@as(usize, 0), f.live);
    try std.testing.expectEqual(@as(usize, 0), s.entries.items.len);
}

test "empty session save reports empty and a missing file never silently becomes a cold conversation" {
    const a = std.testing.allocator;
    var f: Fake = .{ .gpa = a };
    var s = pc.Store.init(a, f.snapshots(), rules, 1 << 30);
    defer s.deinit();
    s.session_identity = 1;
    const got = try sf.save(&s, "/tmp/nonexistent-session-dir/a.tfs");
    try std.testing.expectEqual(@as(u32, 0), got.tokens);
    try std.testing.expectEqual(@as(u64, 0), got.bytes);
    try std.testing.expectError(error.RowsMissing, sf.load(&s, "/tmp/nonexistent-session-dir/a.tfs"));
}

test "session import rejects changed arithmetic, metadata corruption and a truncated commit" {
    const a = std.testing.allocator;
    var buf: [128]u8 = undefined;
    const p = path(&buf, "metadata");
    defer _ = sys.c.unlink(p);
    const toks = [_]u32{ 1, 2, 3, 4, 5, 6, 7 };
    var f: Fake = .{ .gpa = a, .spans = &.{.{ 2, 4 }} };
    var s = pc.Store.init(a, f.snapshots(), rules, 1 << 30);
    defer s.deinit();
    s.session_identity = 55;
    try populate(&s, &f, &toks);
    _ = try sf.save(&s, p);
    _ = try sf.clear(&s);
    s.rules.chunk = 256;
    try std.testing.expectError(error.SessionFormat, sf.load(&s, p));
    s.rules.chunk = 128;
    _ = try sf.load(&s, p);
    try std.testing.expectError(error.SessionPrefixMismatch, s.begin(a, &toks, 6, &.{}, &.{}, null, &.{}));
    _ = try sf.clear(&s);
    var opened = try @import("rows_file.zig").open(p, .state, null);
    const offset = opened.trailer.payload + @sizeOf(u32);
    opened.close();
    const fd = sys.c.open(p, .{ .ACCMODE = .RDWR, .NOFOLLOW = true }, @as(sys.c.mode_t, 0));
    try std.testing.expect(fd >= 0);
    defer _ = sys.c.close(fd);
    try std.testing.expectEqual(@as(isize, 1), sys.c.pwrite(fd, "Q", 1, @intCast(offset)));
    try std.testing.expectError(error.RowsHash, sf.load(&s, p));
    try std.testing.expectEqual(@as(usize, 0), f.live);
    const end = sys.c.lseek(fd, 0, sys.c.SEEK.END);
    try std.testing.expectEqual(@as(c_int, 0), sys.c.ftruncate(fd, end - 1));
    try std.testing.expectError(error.RowsRead, sf.load(&s, p));
}

test "a corrupt commit with overflowing payload size is rejected without an arithmetic trap" {
    const a = std.testing.allocator;
    var buf: [128]u8 = undefined;
    const p = path(&buf, "overflow");
    defer _ = sys.c.unlink(p);
    const toks = [_]u32{ 1, 2, 3, 4, 5, 6 };
    var f: Fake = .{ .gpa = a };
    var s = pc.Store.init(a, f.snapshots(), rules, 1 << 30);
    defer s.deinit();
    s.session_identity = 1;
    try populate(&s, &f, &toks);
    const saved = try sf.save(&s, p);
    _ = try sf.clear(&s);
    const fd = sys.c.open(p, .{ .ACCMODE = .RDWR, .NOFOLLOW = true }, @as(sys.c.mode_t, 0));
    try std.testing.expect(fd >= 0);
    defer _ = sys.c.close(fd);
    const huge: u64 = std.math.maxInt(u64);
    try std.testing.expectEqual(@as(isize, 8), sys.c.pwrite(fd, std.mem.asBytes(&huge).ptr, 8, @intCast(saved.bytes - 96 + 16)));
    try std.testing.expectError(error.RowsRead, sf.load(&s, p));
    try std.testing.expectEqual(@as(usize, 0), f.live);
}
